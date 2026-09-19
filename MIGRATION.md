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

## Version 0.2 — lanceur train/test

- `configs/config.yaml` devient l’entrée principale ; l’ancien `pipeline.yaml`
  reste un exemple expérimental réservé aux scripts numérotés.
- `run.py`, `python -m coolpath` et la commande `coolpath` dispatchent `train`
  et `test`. Précontrôles, sorties isolées et suivi du statut sont ajoutés.
- Les chemins relatifs sont résolus depuis la racine configurée, et non `src/`.
- Images/masques validés, suffixes mixtes normalisés en PNG pour l’apprentissage.
- Split aléatoire robuste aux noms quelconques, groupes réellement disjoints,
  ou listes explicites. La stratification historique intra-vidéo ne prouve pas
  l’indépendance des lots ; la ligne « grouped by video » ci-dessus désigne cet
  ancien comportement, pas le nouveau mode `group`.
- Configuration MMSeg : remplacement complet des dataloaders hérités pour éviter
  de garder `label_map`/`reduce_zero_label` de Cityscapes ; AMP explicite, seed
  configurable, meilleur checkpoint choisi par itération numérique, export résolu.
- BFMS chargé depuis un dossier Hugging Face avec sa configuration de processeur.
  La convention de post-traitement du notebook est conservée.
- La moyenne RGB de peinture est accumulée par sommes de pixels, avec le traitement
  des pixels quasi noirs du notebook ; consommation mémoire réduite.
- Le nombre de secteurs est transmis au calcul des indicateurs.
- Scores de test indépendant et aperçus colorés ajoutés ; définitions de mIoU
  documentées pour ne pas confondre les conventions.

Les modules historiques ne sont pas une reproduction certifiée des figures et
résultats complets. Voir `VALIDATION.md` et `docs/METHODOLOGIE.md` pour la portée.
