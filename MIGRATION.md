# Notebook → Python migration map

| Notebook | Python modules | Main script |
|---|---|---|
| 22 | `data/video360.py`, `data/preannotation.py` | `01_extract_select_frames.py`, `02_preannotate_cvat.py` |
| 23 | `data/cvat.py`, `segmentation/training.py`, `segmentation/evaluation.py` | `03_prepare_training_dataset.py`, `04_train_mask2former.py`, `05_run_semantic_inference.py` |
| 24 | `materials/bfms.py` | `06_run_bfms.py` |
| 25 | `materials/albedo.py`, `materials/fusion.py` | `07_fuse_materials_albedo.py` |
| 29 | `indicators/geometry.py`, `indicators/morphology.py` | `08_compute_urban_indicators.py` |
| 30 | `thermal/*`, `analysis/*` | `09_analyze_microclimate.py` |

## Preserved experiment choices

- Semantic ontology: Cityscapes 19 classes.
- Training split: grouped by video, seed 42, approximately 80/20 (the supplied dataset produced 93 train / 23 validation).
- Mask2Former: crop 512×1024, batch 2 with accumulation 4, AdamW 5e-5, 5000 iterations, validation every 250 iterations, Swin first two stages frozen.
- BFMS: three-scale inference at 768, 896 and 1024.
- Fusion: semantic gating before albedo assignment; sky/person/rider excluded from albedo.
- Paint: one dataset-level shadow-corrected mean color is mapped once to the nearest RAL value, then a fixed paint albedo is used.
- SVF: spherical solid-angle factor `sin(theta)` and horizontal projection `cos(theta)`.
- GVI: both raw image fraction and solid-angle-weighted form are available.
- Permeability: semantic/material LUT plus the final low-vegetation heuristic (-7° elevation, 12° azimuth width).
- Thermal analysis: Tmrt from globe temperature, UTCI, globe-inertia calibration over tau/lag grids, historical morphology, robust correlations and multiple-testing correction.

## Intentional differences from notebooks

1. All paths are configurable and repository-relative instead of hard-coded Windows paths.
2. The complete-route image location is unified as `paths.all_images_dir`.
3. Interactive plots and ad-hoc diagnostic cells are not executed as part of the production pipeline.
4. Raw data and model weights are excluded from version control.
5. Numerical logic is separated from deep-learning model loading to enable unit testing.
