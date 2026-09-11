# CoolPath 360° pipeline

Python refactoring of the PFE notebooks used to characterize the pedestrian-visible urban environment from 360° panoramas and relate it to microclimatic measurements.

## Scientific pipeline

```text
360° videos
  -> frame extraction / blur filtering / diverse selection
  -> CVAT pre-annotation and manual correction
  -> Mask2Former fine tuning (Cityscapes 19 classes)
  -> semantic inference on the full route
  -> BFMS material segmentation
  -> semantic/material gating + fixed albedo / RAL
  -> spherical-geometry-aware SVF/GVI + permeability + visible building fraction
  -> Comfy'Pack processing + Tmrt + UTCI
  -> globe inertia / historical exposure
  -> Pearson, Spearman, effective N, partial Spearman permutations, FDR, Moran I and HAC models
```

The repository preserves the distinction used in the notebooks: the manually validated annotation set is used for training/validation, while the full set of route frames is used for inference and downstream indicators.

## Repository layout

```text
configs/                 Reproducible parameters and paths
resources/               RAL/albedo lookup extracted from notebook 25
src/coolpath/data/       Video extraction, selection, CVAT conversion/preannotation
src/coolpath/segmentation/ Mask2Former training, inference and evaluation
src/coolpath/materials/  BFMS, semantic gating, RAL and albedo
src/coolpath/indicators/ Equirectangular geometry and urban indicators
src/coolpath/thermal/    Comfy'Pack QC, Tmrt, UTCI and thermal history
src/coolpath/analysis/   Statistical analysis
scripts/                 Ordered command-line entry points
tests/                   Unit tests for core numerical logic
```

## Important refactoring decision

The notebooks used two aliases for the directory containing the complete route images (`video360_annotation/candidate_frames` and `dataset_coolpath_360/candidate_frames`). The refactored code uses one configurable path only: `paths.all_images_dir`. The default points to the directory actually produced by notebook 22.

No raw data, model checkpoint, CVAT export or BFMS weights are included. Put them at the paths configured in `configs/pipeline.yaml` or edit the YAML for your machine. This removes the Windows absolute paths embedded in the notebooks.

## Installation

Python 3.10 is recommended because it matches the notebook environment. Create an environment and install the package:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

PyTorch/MMCV are CUDA-sensitive. Install the PyTorch build matching your NVIDIA driver/CUDA first, then install the compatible MMCV build for MMSegmentation 1.2.2. The notebooks were run with MMSegmentation 1.2.2; different notebook stages used different PyTorch/CUDA builds, so PyTorch is deliberately not hard-pinned here.

## Configuration

Edit `configs/pipeline.yaml`. In particular:

- `paths.raw_videos_dir`
- `paths.cvat_export_dir`
- `paths.checkpoints_dir`
- `paths.bfms_model_dir`
- `paths.comfypack_csv`

The defaults for the main experimental parameters are copied from the supplied notebooks, including 0.5 candidate fps, blur threshold 100, 5000 Mask2Former iterations, BFMS scales 768/896/1024, vegetation elevation threshold -7°, azimuth-width filter 12°, and 30/60/120 s analysis windows.

## Run the pipeline

```bash
python scripts/01_extract_select_frames.py
# optional automatic CVAT polygon preannotation, using an already available semantic checkpoint:
python scripts/02_preannotate_cvat.py --model-config PATH_TO_CFG --checkpoint PATH_TO_PTH

# After manual correction/export from CVAT:
python scripts/03_prepare_training_dataset.py
python scripts/04_train_mask2former.py
python scripts/05_run_semantic_inference.py
python scripts/06_run_bfms.py
python scripts/07_fuse_materials_albedo.py
python scripts/08_compute_urban_indicators.py
python scripts/09_analyze_microclimate.py
```

## What was intentionally cleaned up

Notebook-only displays, one-off debug cells and repeated diagnostic plots were not mixed into the production functions. The scientifically relevant computations were kept in modules. Model/data paths and experiment parameters were moved to YAML. Expensive deep-learning imports are local to the functions that need them so the numerical modules can be tested without loading CUDA models.

The BFMS semantic gating remains a pragmatic consistency correction. It should not be interpreted as validation of the raw BFMS predictions on the CoolPath domain.

## Tests

```bash
pytest -q
```

The tests cover analytical SVF cases (full sky and 30°/45°/60° zenith cones), solid-angle GVI behavior, semantic/material gating, FDR and correlation calculations.

## Push to GitHub

```bash
git init
git add .
git commit -m "Refactor CoolPath notebooks into reproducible Python pipeline"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

Before pushing, verify that no raw measurements or large model weights were accidentally added (`git status`). The supplied `.gitignore` excludes those by default.
