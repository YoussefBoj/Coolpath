# Modèles à ajouter localement

- `semantic/` : configuration d’inférence résolue + poids sémantiques.
- `pretrained/` : modèle complet Mask2Former Swin-B Cityscapes de départ.
- `bfms/` : dossier Hugging Face avec poids, config et processeur matériaux.

Lire [le guide modèles](../docs/MODELES.md). Les poids ne sont pas inclus et sont
ignorés par Git. Un entraînement produit son modèle dans `outputs/train_.../model/`.
