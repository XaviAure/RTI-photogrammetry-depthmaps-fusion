# SPDX-License-Identifier: GPL-3.0-or-later
"""
Photogrammetry + RTI depth map fusion for tactile heritage reproductions.

Merges depth maps from photogrammetry (overall shape) and RTI/photometric
stereo (surface detail) into a single combined depth map. Uses image pyramid
processing to separate shape from detail, then recombines the best of each.

Intended for producing tactile reproductions of heritage objects via CNC
milling, where both accurate shape and fine surface texture are needed.

License: GNU General Public License v3.0 or later (GPL-3.0-or-later)

Usage:
    python PG_PS_depthmap_merging.py --ps_folder PS_depthmaps --pg_folder PG_depthmaps \\
        --output_folder merged --levels 3 --detail_percent 12
"""

import numpy as np
import cv2
import os
import re
from osgeo import gdal
import argparse
import logging
from contextlib import contextmanager
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

PG_BACKGROUND = 65535
PS_BACKGROUND = 0


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
    original_shape = image.shape

    # Downsample N times (or until the image cannot be downsampled further)
    current = image.astype(np.float32)
    actual_levels = 0
    for _ in range(levels):
        if min(current.shape[:2]) < 2:
            break
        current = cv2.pyrDown(current)
        actual_levels += 1

    # Upsample back to original size using the number of successful downsamples
    for _ in range(actual_levels):
        current = cv2.pyrUp(current)

    # pyrUp might not give exact original size due to rounding
    # Resize to exact original dimensions if needed
    if current.shape != original_shape:
        current = cv2.resize(current, (original_shape[1], original_shape[0]),
                            interpolation=cv2.INTER_LINEAR)

    return current


def process_image_pair(ps_map, pg_map, levels, global_scale_factor):
    """
    Merges RTI/PS surface detail onto photogrammetry shape.

    Uses a single global_scale_factor so detail strength is consistent
    across all images in the set.
    """
    pg_valid_mask = pg_map != PG_BACKGROUND
    ps_valid_mask = ps_map != PS_BACKGROUND
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

    # Combine PG shape + scaled PS detail
    enhanced_map = low_freq_pg + high_freq_scaled
    enhanced_map[~output_mask] = PG_BACKGROUND

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
                        help="Max PS detail as %% of PG range")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    if args.levels < 0:
        raise ValueError("--levels must be >= 0")
    if args.detail_percent < 0:
        raise ValueError("--detail_percent must be >= 0")

    logging.info("=" * 70)
    logging.info("PG + PS DEPTH MAP FUSION")
    logging.info("=" * 70)
    logging.info(f"Pyramid levels: {args.levels} ({2**args.levels}x downscale)")
    logging.info(f"Detail percent: ±{args.detail_percent}%")
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

    for ps_f, pg_f in tqdm(zip(ps_filenames, pg_filenames), total=len(ps_filenames),
                           desc="Analyzing", disable=args.no_progress):
        # Load maps
        ps_map = load_depth_map(os.path.join(args.ps_folder, ps_f))
        pg_map = load_depth_map(os.path.join(args.pg_folder, pg_f))

        # PG range
        pg_valid = pg_map[pg_map != PG_BACKGROUND]
        if len(pg_valid) > 0:
            global_pg_min = min(global_pg_min, np.min(pg_valid))
            global_pg_max = max(global_pg_max, np.max(pg_valid))

        # Measure PS detail strength (extract detail first)
        pg_valid_mask = pg_map != PG_BACKGROUND
        ps_valid_mask = ps_map != PS_BACKGROUND
        output_mask = pg_valid_mask & ps_valid_mask

        ps_for_filter = ps_map.copy()
        if np.any(~ps_valid_mask):
            ps_mean = np.mean(ps_map[ps_valid_mask]) if np.any(ps_valid_mask) else 0
            ps_for_filter[~ps_valid_mask] = ps_mean

        low_freq_ps = pyramid_lowpass(ps_for_filter, args.levels)
        high_freq_ps = ps_map - low_freq_ps

        if np.any(output_mask):
            ps_high_max = np.max(np.abs(high_freq_ps[output_mask]))
            global_ps_high_max = max(global_ps_high_max, ps_high_max)

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
    logging.info(f"Global PS detail max: {global_ps_high_max:.0f}")
    logging.info(f"Global scale factor: {global_scale_factor:.6f}")

    # =========================================================================
    # PASS 2: Process all images with consistent global scale factor
    # =========================================================================
    logging.info("Pass 2: Merging with global scale factor...")

    for ps_f, pg_f in tqdm(zip(ps_filenames, pg_filenames), total=len(ps_filenames),
                           desc="Merging", disable=args.no_progress):
        try:
            ps_map = load_depth_map(os.path.join(args.ps_folder, ps_f))
            pg_map = load_depth_map(os.path.join(args.pg_folder, pg_f))

            if ps_map.shape != pg_map.shape:
                logging.error(f"Size mismatch: {ps_f}")
                continue

            enhanced = process_image_pair(
                ps_map, pg_map,
                args.levels, global_scale_factor
            )

            enhanced = np.clip(enhanced, 0, 65535).astype(np.uint16)
            output_path = os.path.join(args.output_folder, build_output_filename(pg_f))
            save_depth_map(output_path, enhanced)

        except Exception as e:
            logging.error(f"Error processing {ps_f}: {e}")

    logging.info("=" * 70)
    logging.info("Done! All images merged with consistent detail scaling.")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
