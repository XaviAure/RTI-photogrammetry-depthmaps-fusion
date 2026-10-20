# SPDX-License-Identifier: GPL-3.0-or-later
"""
Photogrammetry + RTI depth map fusion for tactile heritage reproductions.

Merges depth maps from photogrammetry (overall shape) and RTI/photometric
stereo (surface detail) into a single combined depth map. Uses image pyramid
processing to separate shape from detail, then recombines the best of each.

Intended for producing tactile reproductions of heritage objects via CNC
milling, where both accurate shape and fine surface texture are needed.

Method note: shape and detail are separated with a Gaussian-pyramid low-pass
(cv2.pyrDown/pyrUp; Burt & Adelson, 1983). Taking low frequencies from a
position measurement and high frequencies from photometric normals follows
the idea of Nehab et al. (SIGGRAPH 2005). See the references in the README.

Orientation: PS and PG maps must use the same depth direction. PG maps encode
depth_m = global_max - value * precision (larger value = closer to camera).
If the merged detail looks inverted, re-run BiNI with/without --invert.

NoData conventions (important for mixed-source inputs):
    PG depth maps are expected to use uint16 == 65535 as the NoData sentinel.
    PS depth maps may use either 65535 (canonical, modern BiNI output) or 0
    (legacy photometric-stereo tools). The default --ps_background=auto first
    reads the sidecar written by bini-depth-pyramid-gpu.py (<stem>_metadata.txt
    or <stem>.json next to the PS TIFF); without one it guesses per file from
    the histogram and alignment with PG NoData.
    Override with --ps_background 0 or 65535 if auto-detection picks wrong.

License: GNU General Public License v3.0 or later (GPL-3.0-or-later)

Usage:
    python PG_PS_depthmap_merging.py --ps_folder PS_depthmaps --pg_folder PG_depthmaps \\
        --output_folder merged --levels 3 --detail_percent 5
"""

import numpy as np
import cv2
import os
import re
import sys
import json
from osgeo import gdal
import argparse
import logging
from contextlib import contextmanager
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

PG_BACKGROUND = 65535
PS_BACKGROUND_CANONICAL = 65535  # The convention this script and modern BiNI use
PS_BACKGROUND_LEGACY = 0         # Older PS / photometric-stereo tools


def detect_ps_background(ps_map, pg_map=None):
    """
    Auto-detect whether a PS depth map uses 0 or 65535 as its NoData sentinel.

    Strategy, in order:
      1. Compare cluster sizes at 0 and 65535. The NoData sentinel will dominate
         by 10:1 or more over any legitimate extremal valid pixels.
      2. If neither value dominates (both appear in similar amounts), align with
         PG's NoData mask: wherever PG is invalid the matching PS pixel must
         also be invalid, so whichever sentinel value shows up at those
         positions is the correct one.
      3. Default to the canonical 65535 if still ambiguous.

    Returns: int, either 0 or 65535.
    """
    n_zeros = int((ps_map == 0).sum())
    n_maxes = int((ps_map == 65535).sum())

    if n_maxes > 10 * max(n_zeros, 1):
        return PS_BACKGROUND_CANONICAL
    if n_zeros > 10 * max(n_maxes, 1):
        return PS_BACKGROUND_LEGACY

    if pg_map is not None:
        pg_nodata = pg_map == PG_BACKGROUND
        if pg_nodata.any():
            ps_max_at_pg_nd = int((pg_nodata & (ps_map == 65535)).sum())
            ps_zero_at_pg_nd = int((pg_nodata & (ps_map == 0)).sum())
            # Symmetric strong-signal detection: require >2x margin in either
            # direction. Misclassification cost is symmetric (the wrong NoData
            # sentinel leaks into the merge as garbage), so neither convention
            # gets a head start in close-call cases.
            if ps_max_at_pg_nd > 2 * ps_zero_at_pg_nd and ps_max_at_pg_nd > 0:
                return PS_BACKGROUND_CANONICAL
            if ps_zero_at_pg_nd > 2 * ps_max_at_pg_nd and ps_zero_at_pg_nd > 0:
                return PS_BACKGROUND_LEGACY

    # Genuinely ambiguous (no strong signal either way) → canonical default,
    # which is the project's preferred convention going forward.
    return PS_BACKGROUND_CANONICAL


def ps_background_from_sidecar(ps_path):
    """Return 65535 if the sidecar written by bini-depth-pyramid-gpu.py says
    65535 is NoData, otherwise None (no sidecar, or legacy output without NoData)."""
    if ps_path is None:
        return None
    stem = os.path.splitext(ps_path)[0]
    txt_path = stem + '_metadata.txt'
    if os.path.exists(txt_path):
        with open(txt_path, errors='replace') as f:
            if 'reserved as NoData' in f.read():
                return PS_BACKGROUND_CANONICAL
    json_path = stem + '.json'
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                if 'precision_m_per_value' in json.load(f):
                    return PS_BACKGROUND_CANONICAL
        except (ValueError, OSError):
            pass
    return None


def resolve_ps_background(setting, ps_map, pg_map=None, ps_path=None):
    """Translate --ps_background CLI value ('auto' | '0' | '65535') to an int."""
    if setting == 'auto':
        from_sidecar = ps_background_from_sidecar(ps_path)
        if from_sidecar is not None:
            return from_sidecar
        return detect_ps_background(ps_map, pg_map)
    return int(setting)


@contextmanager
def gdal_open(filename, mode):
    ds = gdal.Open(filename, mode)
    if ds is None:
        raise IOError(f"Could not open {filename}")
    try:
        yield ds
    finally:
        ds = None


def load_depth_map(filename):
    with gdal_open(filename, gdal.GA_ReadOnly) as ds:
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
    return array


def save_depth_map(filename, array, nodata_value=65535):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    dataset = driver.Create(filename, cols, rows, 1, gdal.GDT_UInt16)
    if dataset is None:
        raise IOError(f"Could not create {filename}")
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata_value)
    dataset.FlushCache()
    dataset = None


def pyramid_lowpass(image, levels):
    """
    Extract the overall shape (low-detail component) by shrinking and
    re-enlarging the image through a pyramid.

    Args:
        image: Input depth map
        levels: How many times to halve the image before enlarging back.
                More levels = smoother result (3 is a good default).

    Returns:
        Smoothed shape-only version of the input.
    """
    current = image.astype(np.float32)

    # Pad with an odd (point) reflection so slopes carry on past the image
    # edge. A plain mirror bends them and biases the shape near the border,
    # which shows up as a mismatch where neighbouring images overlap.
    pad = 4 * 2 ** levels
    pad_h = min(pad, current.shape[0] - 1)
    pad_w = min(pad, current.shape[1] - 1)
    current = np.pad(current, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect', reflect_type='odd')

    # Downsample N times (or until the image cannot be downsampled further),
    # remembering each size so odd dimensions are restored exactly
    sizes = []
    for _ in range(levels):
        if min(current.shape[:2]) < 2:
            break
        sizes.append((current.shape[1], current.shape[0]))
        current = cv2.pyrDown(current)

    # Upsample back through the same sizes (no stretching on odd dimensions)
    for size in reversed(sizes):
        current = cv2.pyrUp(current, dstsize=size)

    return current[pad_h:current.shape[0] - pad_h, pad_w:current.shape[1] - pad_w]


def process_image_pair(ps_map, pg_map, levels, global_scale_factor, ps_background):
    """
    Merges RTI/PS surface detail onto photogrammetry shape.

    Uses a single global_scale_factor so detail strength is consistent
    across all images in the set. ps_background is the NoData sentinel for the
    PS map (typically 65535 for canonical / new BiNI output, 0 for legacy tools).
    """
    pg_valid_mask = pg_map != PG_BACKGROUND
    ps_valid_mask = ps_map != ps_background
    output_mask = pg_valid_mask & ps_valid_mask

    # Fill backgrounds for cleaner filtering
    pg_for_filter = pg_map.copy()
    if np.any(~pg_valid_mask):
        pg_mean = np.mean(pg_map[pg_valid_mask]) if np.any(pg_valid_mask) else 0
        pg_for_filter[~pg_valid_mask] = pg_mean

    ps_for_filter = ps_map.copy()
    if np.any(~ps_valid_mask):
        ps_mean = np.mean(ps_map[ps_valid_mask]) if np.any(ps_valid_mask) else 0
        ps_for_filter[~ps_valid_mask] = ps_mean

    # Extract overall shape from each source
    low_freq_pg = pyramid_lowpass(pg_for_filter, levels)
    low_freq_ps = pyramid_lowpass(ps_for_filter, levels)

    # Extract surface detail from PS (what's left after removing the shape)
    high_freq_ps = ps_map - low_freq_ps

    # Scale the detail to a consistent strength across the whole set
    high_freq_scaled = high_freq_ps * global_scale_factor

    # Combine PG shape + scaled PS detail. Mark NoData with NaN so the caller
    # can distinguish "saturated valid" from "missing" unambiguously at encode
    # time — the caller clips valid pixels to [0, 65534] and writes 65535 only
    # where the array is NaN.
    enhanced_map = low_freq_pg + high_freq_scaled
    enhanced_map[~output_mask] = np.nan

    return enhanced_map


def get_matched_file_pairs(ps_folder, pg_folder):
    ps_filenames = [f for f in os.listdir(ps_folder) if f.lower().endswith(('.tif', '.tiff'))]
    pg_filenames = [f for f in os.listdir(pg_folder)
                    if f.lower().endswith(('.tif', '.tiff'))
                    and '_mask' not in f.lower()]

    def extract_camera_key(name):
        stem = os.path.splitext(name)[0].lower()
        if stem.endswith('_depth'):
            stem = stem[:-6]

        # Match final camera id token like 1, 27, 27b.
        match = re.search(r'(\d+[a-z]?)$', stem)
        if not match:
            return None
        return match.group(1)

    def build_key_map(filenames, source_name):
        key_map = {}
        skipped = []
        for name in filenames:
            key = extract_camera_key(name)
            if key is None:
                skipped.append(name)
                continue
            if key in key_map:
                raise ValueError(
                    f"Duplicate camera key '{key}' in {source_name}: "
                    f"'{key_map[key]}' and '{name}'"
                )
            key_map[key] = name

        if skipped:
            logging.warning(
                "%s files skipped (could not extract numeric camera key): %s",
                source_name,
                ", ".join(sorted(skipped)[:5])
            )

        return key_map

    def camera_key_sort(key):
        match = re.match(r'^(\d+)([a-z]?)$', key)
        if not match:
            return float('inf'), key
        return int(match.group(1)), match.group(2)

    ps_map = build_key_map(ps_filenames, "PS")
    pg_map = build_key_map(pg_filenames, "PG")

    common_keys = sorted(set(ps_map.keys()) & set(pg_map.keys()), key=camera_key_sort)
    if not common_keys:
        ps_keys = sorted(ps_map.keys(), key=camera_key_sort)
        pg_keys = sorted(pg_map.keys(), key=camera_key_sort)
        raise ValueError(
            "No matching files found between PS and PG folders after camera-key "
            f"normalization. Sample PS keys: {ps_keys[:10]}, PG keys: {pg_keys[:10]}"
        )

    return [ps_map[k] for k in common_keys], [pg_map[k] for k in common_keys]


def build_output_filename(pg_filename):
    stem = os.path.splitext(pg_filename)[0].lower()
    if stem.endswith('_depth'):
        stem = stem[:-6]

    match = re.search(r'(\d+[a-z]?)$', stem)
    if match:
        return f"RTI_{match.group(1)}.tif"

    return os.path.splitext(pg_filename)[0] + ".tif"


def main():
    parser = argparse.ArgumentParser(
        description="Merge photogrammetry and RTI/photometric stereo depth maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ps_folder", type=str, default="PS_depthmaps")
    parser.add_argument("--pg_folder", type=str, default="PG_depthmaps")
    parser.add_argument("--output_folder", type=str, default="merged_pyramid")
    parser.add_argument("--levels", type=int, default=3,
                        help="Pyramid levels. Each level = 2x downscale. "
                             "3 levels = 8x smaller = smooth low-freq.")
    parser.add_argument("--detail_percent", type=float, default=5.0,
                        help="Detail strength: the 99.5th percentile of |PS detail| over the "
                             "whole set is scaled to this %% of the PG depth range")
    parser.add_argument("--scale_factor", type=float, default=None,
                        help="Use this detail scale factor instead of computing one from "
                             "--detail_percent. Pass the 'Global scale factor' printed by an "
                             "earlier run to merge more images of the same object with "
                             "matching detail strength.")
    parser.add_argument("--ps_background", choices=['auto', '0', '65535'], default='auto',
                        help="NoData sentinel for PS depth maps. 'auto' detects per-file "
                             "from the histogram and PG alignment (recommended). "
                             "Use '65535' for canonical BiNI output, '0' for legacy PS tools.")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    if args.levels < 0:
        raise ValueError("--levels must be >= 0")
    if args.detail_percent < 0:
        raise ValueError("--detail_percent must be >= 0")
    if args.scale_factor is not None and args.scale_factor < 0:
        raise ValueError("--scale_factor must be >= 0")

    logging.info("=" * 70)
    logging.info("PG + PS DEPTH MAP FUSION")
    logging.info("=" * 70)
    logging.info(f"Pyramid levels: {args.levels} ({2**args.levels}x downscale)")
    logging.info(f"Detail percent: +/-{args.detail_percent}%")
    logging.info("=" * 70)

    os.makedirs(args.output_folder, exist_ok=True)

    ps_filenames, pg_filenames = get_matched_file_pairs(args.ps_folder, args.pg_folder)
    logging.info(f"Found {len(ps_filenames)} image pairs")

    # =========================================================================
    # PASS 1: Measure depth ranges and detail strength across all images
    # =========================================================================
    logging.info("Pass 1: Measuring depth ranges and detail strength...")

    global_pg_min = np.inf
    global_pg_max = -np.inf
    global_ps_high_max = 0
    ps_background_by_file = {}
    skipped = set()

    for ps_f, pg_f in tqdm(zip(ps_filenames, pg_filenames), total=len(ps_filenames),
                           desc="Analyzing", disable=args.no_progress):
        # Load maps
        ps_path = os.path.join(args.ps_folder, ps_f)
        ps_map = load_depth_map(ps_path)
        pg_map = load_depth_map(os.path.join(args.pg_folder, pg_f))

        if ps_map.shape != pg_map.shape:
            logging.error(f"Size mismatch, skipping: {ps_f} {ps_map.shape} vs {pg_f} {pg_map.shape}")
            skipped.add(ps_f)
            continue

        # Resolve PS background sentinel for this pair (and cache for Pass 2)
        ps_background = resolve_ps_background(args.ps_background, ps_map, pg_map, ps_path)
        ps_background_by_file[ps_f] = ps_background
        if args.ps_background == 'auto':
            logging.info(f"  {ps_f}: detected PS NoData sentinel = {ps_background}")

        # PG range
        pg_valid = pg_map[pg_map != PG_BACKGROUND]
        if len(pg_valid) > 0:
            global_pg_min = min(global_pg_min, np.min(pg_valid))
            global_pg_max = max(global_pg_max, np.max(pg_valid))

        # Measure PS detail strength (extract detail first)
        pg_valid_mask = pg_map != PG_BACKGROUND
        ps_valid_mask = ps_map != ps_background
        output_mask = pg_valid_mask & ps_valid_mask

        ps_for_filter = ps_map.copy()
        if np.any(~ps_valid_mask):
            ps_mean = np.mean(ps_map[ps_valid_mask]) if np.any(ps_valid_mask) else 0
            ps_for_filter[~ps_valid_mask] = ps_mean

        low_freq_ps = pyramid_lowpass(ps_for_filter, args.levels)
        high_freq_ps = ps_map - low_freq_ps

        if np.any(output_mask):
            # Use the 99.5th percentile of |high_freq| rather than the absolute
            # max so a single bright outlier in one file can't squash detail
            # across the entire batch.
            ps_high_robust = float(np.percentile(np.abs(high_freq_ps[output_mask]), 99.5))
            global_ps_high_max = max(global_ps_high_max, ps_high_robust)

    if np.isfinite(global_pg_min) and np.isfinite(global_pg_max):
        global_pg_range = global_pg_max - global_pg_min
    else:
        logging.warning("No valid PG pixels found; forcing global_pg_range to 0")
        global_pg_min = 0
        global_pg_max = 0
        global_pg_range = 0

    # Compute single global scale factor
    target_max = (args.detail_percent / 100.0) * global_pg_range
    if global_ps_high_max > 0:
        global_scale_factor = target_max / global_ps_high_max
    else:
        global_scale_factor = 1.0

    logging.info(f"Global PG range: {global_pg_min:.0f} - {global_pg_max:.0f} = {global_pg_range:.0f}")
    logging.info(f"Global PS detail (p99.5 of |high_freq|): {global_ps_high_max:.0f}")
    if args.scale_factor is not None:
        logging.info(f"Using --scale_factor {args.scale_factor:.6f} "
                     f"(--detail_percent would give {global_scale_factor:.6f})")
        global_scale_factor = args.scale_factor
    logging.info(f"Global scale factor: {global_scale_factor:.6f}")

    if ps_background_by_file:
        canonical_count = sum(1 for v in ps_background_by_file.values() if v == 65535)
        legacy_count = sum(1 for v in ps_background_by_file.values() if v == 0)
        logging.info(
            f"PS NoData sentinel: {canonical_count} files at 65535 (canonical), "
            f"{legacy_count} at 0 (legacy)"
        )

    # =========================================================================
    # PASS 2: Process all images with consistent global scale factor
    # =========================================================================
    logging.info("Pass 2: Merging with global scale factor...")

    n_done = 0
    n_failed = 0
    for ps_f, pg_f in tqdm(zip(ps_filenames, pg_filenames), total=len(ps_filenames),
                           desc="Merging", disable=args.no_progress):
        if ps_f in skipped:
            continue
        try:
            ps_path = os.path.join(args.ps_folder, ps_f)
            ps_map = load_depth_map(ps_path)
            pg_map = load_depth_map(os.path.join(args.pg_folder, pg_f))

            ps_background = ps_background_by_file.get(ps_f)
            if ps_background is None:
                ps_background = resolve_ps_background(args.ps_background, ps_map, pg_map, ps_path)
            enhanced = process_image_pair(
                ps_map, pg_map,
                args.levels, global_scale_factor, ps_background
            )

            # Encode to uint16: valid pixels clipped to [0, 65534], NoData at 65535.
            # process_image_pair marks NoData with NaN; testing for NaN here means
            # a saturated valid pixel (one that happens to land exactly on 65535
            # from the merge math) can never be misclassified as NoData.
            nodata_mask = np.isnan(enhanced)
            encoded = np.full(enhanced.shape, PG_BACKGROUND, dtype=np.uint16)
            valid = ~nodata_mask
            encoded[valid] = np.clip(enhanced[valid], 0, 65534).astype(np.uint16)

            output_path = os.path.join(args.output_folder, build_output_filename(pg_f))
            save_depth_map(output_path, encoded)
            n_done += 1

        except Exception as e:
            logging.error(f"Error processing {ps_f}: {e}")
            n_failed += 1

    logging.info("=" * 70)
    logging.info(f"Done: {n_done} merged, {len(skipped)} skipped (size mismatch), {n_failed} failed.")
    logging.info("=" * 70)
    return 0 if (n_failed == 0 and not skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
