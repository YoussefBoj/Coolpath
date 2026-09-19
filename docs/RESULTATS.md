# Lire les résultats

## Fichiers communs

| Fichier | Contenu |
|---|---|
| `config_used.yaml` | Paramètres effectifs, y compris le mode remplacé en ligne de commande |
| `environment.json` | Versions disponibles des principales bibliothèques |
| `status.json` | `running`, `completed` ou `failed` |
| `summary.json` | Résumé d’un calcul terminé |
| `error.log` | Trace technique lorsqu’un calcul échoue après création du dossier |

Un arrêt brutal de l’ordinateur peut laisser `running`. Un dossier partiel n’est
pas un résultat complet : vérifier `status.json`.

## Mode train

| Chemin | Usage |
|---|---|
| `dataset/images/`, `dataset/labels/` | Copies normalisées des données utilisées |
| `dataset/splits/train.txt`, `val.txt` | Répartition exacte des images |
| `training/` | Configuration et logs MMSegmentation, checkpoints périodiques |
| `model/model.pth` | Checkpoint sélectionné pour l’inférence |
| `model/model_config.py` | Configuration d’inférence avec héritage résolu |

Le checkpoint `best_mIoU` est préféré. À défaut, un checkpoint final disponible
est exporté ; son nom source est consigné dans `summary.json`. Le score de
validation sert à choisir le modèle : ce n’est pas un score de test indépendant.

## Mode test

| Chemin | Condition | Usage |
|---|---|---|
| `semantic/masks/*_sem.png` | Toujours | Indices de classes 0–18 |
| `semantic/previews/*_overlay.jpg` | `previews: true` | Contrôle visuel superposé à l’image |
| `semantic/previews/legende.csv` | `previews: true` | Correspondance indice/classe |
| `evaluation/metrics.json` | Annotations fournies | Scores globaux |
| `evaluation/scores_par_classe.csv` | Annotations fournies | IoU par classe |
| `evaluation/confusion.csv` | Annotations fournies | Lignes = vérité terrain, colonnes = prédiction |
| `bfms/masks_bfms/` | Matériaux | Classes matériaux brutes |
| `bfms/bfms_par_image.csv` | Matériaux | Groupes matériaux par image |
| `fusion/masks_fused/` | Matériaux | Matériaux corrigés par les règles sémantiques |
| `fusion/albedo/` | Matériaux | Cartes numériques `.npy`, NaN = non estimé |
| `fusion/fusion_par_image.csv` | Matériaux | Proportions, taux de correction et albédo |
| `indicators/indicateurs_par_image.csv` | Indicateurs | SVF, végétation, bâti et perméabilité |
| `analysis/` | Microclimat | Tables de contrôle, fenêtres temporelles, corrélations et modèles |

## Scores de segmentation

L’IoU mesure le recouvrement entre les pixels prédits et les pixels réellement
annotés : 100 % représente un recouvrement parfait.

- `mIoU_pct` : moyenne sur les classes présentes dans l’union GT/prédiction ;
  une classe prédite à tort participe donc à la moyenne avec une IoU de zéro.
- `mIoU_GT_present_pct` : moyenne limitée aux classes présentes dans les
  annotations, pour comparaison avec la convention du notebook 23.
- `pixel_accuracy_pct` : proportion de pixels annotés correctement classés.

Les pixels 255 sont exclus. Une classe absente de la vérité et des prédictions
a une IoU vide (NaN), pas une IoU de 100 %. Ne pas comparer des mIoU calculées
avec des conventions ou des jeux d’images différents sans le préciser.

## Indicateurs

- **SVF** : valeur de 0 à 1, estimation de l’ouverture au ciel avec pondération
  sphérique et projection horizontale.
- **GVI total** : pourcentage des classes vegetation + terrain, ce qui inclut
  potentiellement du sol nu. Ce n’est donc pas une mesure pure de feuilles vertes.
- **GVI haute/basse** : les colonnes historiques renvoient aux classes vegetation
  et terrain ; elles ne sont pas la séparation géométrique buisson/canopée
  utilisée dans le calcul de perméabilité.
- **`_omega_pct`** : variante pondérée par angle solide ; les pourcentages bruts
  en pixels ne compensent pas la distorsion équirectangulaire.
- **Perméable / imperméable / indéterminé** : proportions sur le sol observable
  retenu par la méthode, pas sur toute la surface réelle du quartier.
- **Canopée, sol non observable** : végétation haute sous laquelle le sol n’est
  pas identifié. Ne pas la convertir automatiquement en sol perméable.
- **`pct_corriges_gating`** : taux de pixels BFMS remplacés par les règles. Un
  taux élevé invite à contrôler visuellement les prédictions matériaux.

Les colonnes thermiques sont documentées par les modules `thermal` et
`analysis`. Les graphiques interactifs des notebooks ne sont pas tous recréés :
les tables numériques sont conservées, avec des aperçus sémantiques ajoutés.
