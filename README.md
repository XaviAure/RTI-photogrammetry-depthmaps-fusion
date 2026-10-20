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
    --detail_percent 5
```

### Parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `--levels` | 3 | Pyramid depth — how much smoothing separates "shape" from "detail". Higher = smoother shape component. |
| `--detail_percent` | 5 | How strong the RTI detail appears in the output, as a percentage of the photogrammetry depth range (the 99.5th percentile of the detail is scaled to this value). |
| `--scale_factor` | none | Use a fixed detail scale factor instead of `--detail_percent`, e.g. the "Global scale factor" printed by an earlier run. |
| `--ps_background` | auto | NoData value of the PS maps: `auto`, `0` or `65535`. |

**Detail strength changed in this version.** The detail is now scaled by its 99.5th percentile rather than its maximum, so a few extreme pixels cannot flatten the detail of the whole set. The same `--detail_percent` therefore gives much stronger detail than the first version of this script (on a 30-image embroidery set, `5` gave a factor of 4.8 instead of 0.31). If the PS maps are metric (same `precision_m_per_value` as the PG maps), `--scale_factor 1` adds the detail at its measured height.

### Blending the outputs back in Metashape

Every image gets the same scale factor, so detail strength matches where images overlap. Merge all images of one object in a single run. To add or redo images later, pass the "Global scale factor" from the first run's log with `--scale_factor`; otherwise the factor is computed from the new images only and will not match. Away from the image edges, overlapping images give identical values; the outer ~16 px of each image cannot match exactly, because the smoothing needs pixels beyond the edge.

## Input files

- **PG folder**: `RTI_<id>_depth.tif` (16-bit TIFF, 65535 = background)
- **PS folder**: `RTI_<id>.tif` or `RTI_<id>.tiff` (16-bit TIFF, same size as the PG map). The background value is read from the `_metadata.txt` / `.json` file written next to each map by the BiNI script (65535); without one it is detected from the image (0 or 65535). Override with `--ps_background`.

## Output files

- `RTI_1.tif`, `RTI_2.tif`, ... — merged 16-bit depth maps (65535 = background)

## Related tools

This script is part of a multi-source depth fusion pipeline:

1. **Export depth maps** from Metashape using [export-depthmaps-from-mesh-Agisoft-Metashape](https://github.com/XaviAure/export-depthmaps-from-mesh-Agisoft-Metashape)
2. **Integrate RTI/photometric stereo normal maps into depth maps** using a modified version of BINI adapted for large images (separate project, coming soon)
3. **Merge PG + PS depth maps** into final outputs (this script)

The RTI or photometric stereo depth maps used as input here are not raw normal maps — they need to be integrated into depth maps first (step 2) before they can be merged with the photogrammetry depth maps.

## References

The shape/detail split is a Gaussian-pyramid low-pass (`cv2.pyrDown` / `cv2.pyrUp`). Taking the low frequencies from a position measurement (photogrammetry) and the high frequencies from photometric normals follows the idea of Nehab et al.; this script is a simple frequency-split version of that idea, not their optimisation.

- Peter J. Burt, Edward H. Adelson. "The Laplacian pyramid as a compact image code." *IEEE Transactions on Communications* COM-31(4):532–540, 1983.
- Diego Nehab, Szymon Rusinkiewicz, James Davis, Ravi Ramamoorthi. "Efficiently combining positions and normals for precise 3D geometry." *ACM Transactions on Graphics (Proc. SIGGRAPH)* 24(3):536–543, 2005.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

