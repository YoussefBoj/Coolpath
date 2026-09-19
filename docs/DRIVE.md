# Données et résultats disponibles sur Google Drive

Le code est sur GitHub. Les vidéos, jeux d’images, annotations, poids entraînés
et résultats volumineux sont distribués séparément sur Google Drive.

**Lien Google Drive : à renseigner par la personne qui publie le dépôt.**
Le lien du Drive n’a pas encore été fourni. Le lien Figshare de BFMS est distinct.

## À quoi sert chaque dossier ?

Les dix noms ci-dessous sont visibles sur la capture fournie. Leur rôle est
reconstitué à partir des notebooks ; le contenu du Drive n’a pas été inspecté.
Les noms internes cités sont donc des repères attendus à vérifier dans l’archive.

| Dossier | Signification et contenu attendu | Quand le télécharger ? |
|---|---|---|
| `cvat_export_360` | Annotations 360° exportées de CVAT : `labelmap.txt`, masques colorés dans `SegmentationClass/`, éventuellement images dans `JPEGImages/`. | Pour réutiliser les annotations d’origine en entraînement. Inutile pour tester des images sans annotations. |
| `raw` | Données brutes d’acquisition : vidéos 360° et/ou mesures Comfy’Pack selon les sous-dossiers réellement fournis. | Pour repartir des acquisitions ou réaliser les analyses thermiques. |
| `video360_annotation` | Préparation des images : frames candidates, sélection à annoter, préannotations et aperçus de contrôle selon les étapes exécutées. Les notebooks utilisent notamment `candidate_frames/` et `frames_to_annotate/`. | Pour retrouver les images du parcours ou préparer les annotations. |
| `final_training_m2f` | Résultats du fine-tuning final Mask2Former : configuration, checkpoints `.pth`, logs, scores et figures selon les sorties conservées. C’est le dossier à examiner pour retrouver le modèle adapté au projet. | Prioritaire pour utiliser le modèle fine-tuné de l’autrice sans réentraîner. |
| `bfms_materials_1041` | Résultats de segmentation matériaux BFMS sur le parcours : notamment `masks_bfms/*_bfms.png` et tableaux de distribution. | Pour consulter les prédictions existantes. Ce dossier de résultats ne remplace pas les poids BFMS. |
| `fusion_3models_v2_1041` | Résultats de combinaison sémantique/matériaux/couleur : masques fusionnés, albédo et tableaux. Peut contenir les masques sémantiques utilisés en aval, selon les cellules exécutées. | Pour consulter la fusion existante ou ses indicateurs. Le nom « 3models » ne désigne pas trois fichiers de poids. |
| `dataset_coolpath_360` | Dataset préparé pour l’apprentissage sémantique : `images/`, `labels/` indexés et `splits/train.txt`, `splits/val.txt`. | Pour réutiliser le dataset annoté préparé et son découpage. |
| `correlation_v2_temporelle_1041` | Dossier d’une analyse temporelle antérieure, d’après son nom. Aucun notebook spécifique à cette version n’a été fourni dans cette mise à jour. | Historique et comparaison ; ne pas le présenter comme la dernière analyse. |
| `correlation_v8_visualisation_complete` | Sorties de l’analyse du notebook 30 : tableaux, corrélations et visualisations, selon les cellules exécutées. | Pour consulter l’analyse la plus récente parmi les notebooks fournis. |
| `indicateurs_v2_1041` | Sorties morphologiques du notebook 29 : SVF, GVI, bâti visible et perméabilité estimée ; notamment `indicateurs_par_image.csv`. | Pour consulter les indicateurs déjà calculés sur le parcours. |

Le suffixe `1041` décrit le parcours historique ; il n’impose pas 1 041 images à
un nouvel utilisateur. Le jeu d’apprentissage annoté et les images complètes du
parcours ont des usages différents : ne pas les confondre.

## Quels fichiers prendre pour mon besoin ?

| Besoin | Minimum nécessaire |
|---|---|
| Tester mes photos avec le modèle de l’autrice | Mes images + un checkpoint sélectionné dans `final_training_m2f` + sa configuration d’inférence portable |
| Calculer aussi matériaux et indicateurs | Le minimum précédent + modèle BFMS complet de Figshare ; panoramas complets 2:1 pour les indicateurs |
| Entraîner sur mes propres annotations | Mes images + mes annotations + configuration et poids de départ Mask2Former ; les données historiques ne sont pas obligatoires |
| Reprendre le dataset annoté original | `dataset_coolpath_360`, ou `cvat_export_360` avec les images correspondantes |
| Refaire les corrélations du parcours original | Images, modèles, mesures de `raw`, dates/heures/FPS corrects ; les sorties historiques permettent une comparaison |
| Consulter uniquement les résultats | Les dossiers de résultats souhaités ; aucun entraînement nécessaire pour lire leurs CSV/images |

## Reconnaître les fichiers du modèle fine-tuné

| Fichier / motif attendu | Ce qu’il signifie |
|---|---|
| `mask2former_coolpath_final.py` | Configuration historique générée par le notebook 23 ; peut contenir des chemins locaux et un héritage `_base_`. |
| `best_mIoU_iter_*.pth` | Checkpoint sauvegardé sur amélioration de la mIoU de validation. Choisir le checkpoint retenu par l’expérience, pas un numéro inventé depuis la capture. |
| `iter_*.pth` | Checkpoint périodique. Le plus récent n’est pas forcément le meilleur sur la validation. |
| `last_checkpoint` | Petit fichier texte pointant vers le dernier checkpoint ; ce n’est pas le modèle lui-même. |
| `*/vis_data/scalars.json` | Mesures de l’apprentissage et de la validation, si conservées. |
| `per_class_metrics_nb23.csv` | Scores par classe produits par le notebook 23, si conservés. |
| `figures/` | Graphiques ; inutiles au chargement du réseau. |

Les noms exacts des `.pth` ne sont pas visibles sur la capture. Ils doivent être
identifiés dans le vrai dossier avant de compléter les chemins du YAML.

## Publier un modèle facile à utiliser

Depuis la machine où la configuration historique et ses fichiers `_base_` sont
encore accessibles, lancer le script d’export fourni :

```bash
python scripts/export_finetuned_model.py --config data/outputs/final_training_m2f/mask2former_coolpath_final.py --checkpoint CHEMIN_VERS_LE_CHECKPOINT_CHOISI.pth --output models/semantic
```

`CHEMIN_VERS_LE_CHECKPOINT_CHOISI.pth` est à remplacer par le vrai chemin.
Le script produit `model_config.py` et `model.pth`. Il résout l’héritage et retire
les paramètres réservés à l’entraînement ainsi que les initialisations externes.
Il ne modifie pas les fichiers sources et refuse d’écraser une destination non vide.

Publier ces deux fichiers ensemble dans un sous-dossier identifiable du Drive,
par exemple `semantic_export/`. L’utilisateur les télécharge dans
`models/semantic/`, puis garde dans `configs/config.yaml` :

```yaml
mode: test
# Dans le bloc test existant, conserver les autres paramètres :
test:
  model_config: models/semantic/model_config.py
  checkpoint: models/semantic/model.pth
  images_dir: data/test/images
```

Ne pas remplacer tout le YAML par ce fragment. Après installation :

```bash
python run.py --check
python run.py
```

Les anciens résultats du Drive ne sont pas automatiquement consommés par le
nouveau lanceur. Celui-ci calcule de nouvelles sorties dans `outputs/test_<date>/`.
Il ne faut donc pas télécharger tous les résultats historiques pour tester ses
propres images.
