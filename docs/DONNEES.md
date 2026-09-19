# Préparer ses données

## Images

JPG/JPEG/PNG, directement dans le dossier indiqué, sans sous-dossiers. Chaque
nom sans extension doit être unique : ne pas mettre à la fois `rue01.jpg` et
`rue01.png`. Les extensions peuvent être différentes d’une image à l’autre ;
le mode train crée des copies PNG RGB normalisées.

La segmentation seule accepte des images ordinaires. Les indicateurs
morphologiques supposent des panoramas équirectangulaires **complets 360° × 180°,
largeur = 2 × hauteur, horizon centré**, et non une photo étirée au ratio 2:1.

## Annotations indexées (option par défaut)

Exemples de paires :

| Image | Annotation |
|---|---|
| `data/train/images/rue01.jpg` | `data/train/labels/rue01.png` |
| `data/train/images/rue02.png` | `data/train/labels/rue02.png` |

Un masque est un PNG **à un canal** : sa valeur est un indice, pas une couleur RGB.
Les masques en mode palette `P` sont acceptés si leurs indices sont bien les
train IDs ci-dessous. Ne pas confondre les IDs Cityscapes originaux avec les
train IDs contigus 0–18. Les dimensions doivent correspondre exactement à l’image.

| Indice | Classe | Sens |
|---|---|---|
| 0 | road | Chaussée |
| 1 | sidewalk | Trottoir |
| 2 | building | Bâtiment |
| 3 | wall | Mur |
| 4 | fence | Clôture |
| 5 | pole | Poteau |
| 6 | traffic_light | Feu de circulation |
| 7 | traffic_sign | Panneau |
| 8 | vegetation | Végétation, dont arbres et haies |
| 9 | terrain | Terrain, herbe ou sol nu |
| 10 | sky | Ciel |
| 11 | person | Personne |
| 12 | rider | Cycliste / conducteur de deux-roues |
| 13 | car | Voiture |
| 14 | truck | Camion |
| 15 | bus | Bus |
| 16 | train | Train |
| 17 | motorcycle | Moto |
| 18 | bicycle | Vélo |
| 255 | ignore | Pixel non annoté, ignoré dans l’apprentissage et les scores |

Un masque entièrement à 255 est refusé. Il faut au moins deux images pour
constituer train et validation ; deux images permettent seulement de vérifier le
fonctionnement, pas d’établir une performance fiable.

## Export CVAT

Utiliser un export **Segmentation mask** décompressé, avec :

| Chemin relatif au dossier d’export | Contenu |
|---|---|
| `labelmap.txt` | Lignes `nom:R,G,B::` |
| `SegmentationClass/nom.png` | Masques colorés |
| `JPEGImages/nom.jpg` | Images, si incluses dans l’export |

Configurer `training.dataset_format: cvat` et `training.cvat_export_dir`.
Si `JPEGImages/` n’est pas présent, les images doivent être dans
`training.images_dir`. Les noms doivent correspondre aux masques.
`background` est converti en 255 ; les autres noms doivent appartenir à la table.
Les espaces de `traffic light` et `traffic sign` sont convertis en underscores.
Les couleurs absentes de `labelmap.txt` restent ignorées, comme dans le notebook.
Un export XML de polygones ne remplace pas cet export raster : le convertir dans
CVAT d’abord. Les préannotations doivent être corrigées manuellement.

## Découper train et validation

| Stratégie | Configuration | Utilisation |
|---|---|---|
| Aléatoire | `split_strategy: random` | Images indépendantes, seed reproductible |
| Groupes séparés | `split_strategy: group`, `groups_csv: data/train/groups.csv` | Vidéos / trajets : un groupe n’apparaît jamais dans les deux lots |
| Listes existantes | `split_strategy: files`, `split_dir: data/train/splits` | Préserver exactement un découpage déjà validé |

Exemple de `groups.csv` (noms d’images **sans extension**) :

```csv
image,group
rue01,parcours_A
rue02,parcours_A
rue03,parcours_B
rue04,parcours_C
```

Pour `files`, créer `train.txt` et `val.txt`, un nom sans extension par ligne.
Les listes doivent être disjointes et couvrir exactement le dataset. Choisir
ces listes aussi pour reprendre un entraînement avec le même découpage.

Avec `group`, la fraction de validation cible un nombre de groupes : la
proportion d’images peut différer de 20 % si les groupes sont de tailles inégales.
Le découpage historique 80/20 **à l’intérieur** de chaque vidéo ne garantit pas
l’indépendance spatiale ou temporelle ; il reste disponible via les scripts
historiques, mais n’est pas présenté comme une séparation par parcours.

## Partir de vidéos 360° (usage avancé)

Les scripts `01_extract_select_frames.py` et `02_preannotate_cvat.py` restent
utilisables. Ils lisent `configs/pipeline.yaml`, pas `configs/config.yaml`.
Modifier les chemins et `selection.quotas` avec vos noms de vidéos réels, puis :

```bash
python scripts/01_extract_select_frames.py --config configs/pipeline.yaml
```

Après sélection, annotation et export CVAT, renseigner ces dossiers dans la
configuration principale. Le lanceur train/test commence aux images : il
n’annotera pas automatiquement un dataset d’entraînement à votre place.

## Mesures thermiques optionnelles

La pipeline thermique conserve le schéma Comfy’Pack du notebook. Un CSV simple,
sans préambule TOA5, peut comporter ces colonnes :

```csv
TIMESTAMP,Prt_Ball_Temperature_Avg,HMP_Temp,HMP_RH,WS_avg,Solspy
2026-06-11 14:22:18,34.0,28.0,52.0,1.0,600.0
```

Ce n’est qu’un exemple de format, **pas un dataset scientifique**. Il faut une
série suffisamment longue et synchronisée. Températures en °C, humidité en %,
vitesse du vent en m/s, rayonnement en W/m². Les noms normalisés `ts,Tg,HMP_Temp,RH,WS,Kdown`
sont aussi acceptés. Garder la même convention d’heure pour images et capteurs.
Le parseur TOA5 hérité est spécifique au format du notebook : pour d’autres
exports, convertir vers ce CSV simple.

Les images utilisées pour l’analyse doivent suivre la convention
`GS017064...__f000120.jpg` (**deux underscores avant f**). Avec
`analysis.t0: {GS017064: '14:22:18'}` et `analysis.fps: {GS017064: 60.0}`,
la frame 120 correspond à 14:22:20. `fps` est la fréquence de la vidéo originale,
pas la fréquence d’extraction (par exemple 0,5 image/s).

Les noms libres sont acceptés pour segmentation, matériaux et indicateurs.
L’analyse temporelle actuelle exige cette convention GoPro ; pour un autre
système de nommage, adapter `timestamp_images` explicitement. Aucun horaire
n’est deviné silencieusement par le lanceur principal.
