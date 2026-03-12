# Photogrammetry + RTI Depth Map Fusion

A Python script that merges depth maps from two different imaging techniques — photogrammetry (PG) and RTI/photometric stereo (PS) — into a single combined depth map. Designed for producing tactile heritage reproductions that capture both overall shape and fine surface detail.

## What it does

The script takes two sets of depth maps of the same object:

- **Photogrammetry depth maps** — accurate overall shape and calibrated scale
- **RTI/photometric stereo depth maps** — fine surface texture and detail

It separates each into low-frequency (shape) and high-frequency (detail) components using image pyramid processing, then combines the photogrammetry shape with the RTI detail. The result is a merged depth map that has the best of both techniques.

Detail strength is scaled consistently across all images in the set, so merged outputs can be used directly for CNC milling or 3D printing without per-image adjustment.

## Why it exists

Neither photogrammetry nor RTI alone captures both the macro geometry and micro surface texture of heritage objects reliably. This script was developed to combine both for:

- CNC milling of tactile reproductions for visually impaired museum visitors
- Any workflow needing both calibrated shape and fine surface detail in a single depth map

## Requirements

- Python 3
- numpy
- opencv-python
- GDAL (osgeo)
- tqdm

See `requirements.txt` for the full list.

## How to use it

Prepare two folders: one with your photogrammetry depth maps, one with your RTI/photometric stereo depth maps. Files are matched automatically by camera number (e.g. `RTI_21.tiff` pairs with `RTI_21_depth.tif`).

```
python PG_PS_depthmap_merging.py \
    --ps_folder PS_depthmaps \
    --pg_folder PG_depthmaps \
    --output_folder merged \
    --levels 3 \
    --detail_percent 12
```

### Parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `--levels` | 3 | Pyramid depth — how much smoothing separates "shape" from "detail". Higher = smoother shape component. |
| `--detail_percent` | 5 | How strong the RTI detail appears in the output, as a percentage of the photogrammetry depth range. Start with 8-12 for heritage objects. |

## Input files

- **PG folder**: `RTI_<id>_depth.tif` (16-bit TIFF, 65535 = background)
- **PS folder**: `RTI_<id>.tif` or `RTI_<id>.tiff` (16-bit TIFF, 0 = background)

## Output files

- `RTI_1.tif`, `RTI_2.tif`, ... — merged 16-bit depth maps (65535 = background)

## Related tools

This script is part of a multi-source depth fusion pipeline:

1. **Export depth maps** from Metashape using [export-depthmaps-from-mesh-Agisoft-Metashape](https://github.com/XaviAure/export-depthmaps-from-mesh-Agisoft-Metashape)
2. **Integrate RTI/photometric stereo normal maps into depth maps** using a modified version of BINI adapted for large images (separate project, coming soon)
3. **Merge PG + PS depth maps** into final outputs (this script)

The RTI or photometric stereo depth maps used as input here are not raw normal maps — they need to be integrated into depth maps first (step 2) before they can be merged with the photogrammetry depth maps.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

