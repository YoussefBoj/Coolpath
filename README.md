# CoolPath — analyser des scènes urbaines à partir d’images

CoolPath transforme des images de rue en cartes des éléments visibles : ciel,
végétation, bâtiments, chaussée, etc. À partir de panoramas 360°, il peut aussi
estimer des indicateurs de végétation, d’ouverture au ciel et de perméabilité
apparente, puis les rapprocher de mesures microclimatiques.

**Deux modes, un seul point d’entrée : `python run.py`.**

| Votre objectif | Mode | Ce que vous apportez | Ce que vous obtenez |
|---|---|---|---|
| Adapter le modèle à vos images | `train` | Images + annotations + modèle de départ | Modèle adapté, configuration exportée, logs et scores de validation |
| Analyser de nouvelles images | `test` | Images + modèle entraîné et sa configuration | Masques, aperçus colorés ; scores si annotations ; analyses optionnelles |

`train` est un **fine-tuning de la segmentation sémantique** sur les 19 classes
Cityscapes du projet. BFMS est utilisé comme modèle matériaux préentraîné ;
son entraînement n’est pas implémenté dans les notebooks fournis.
`test` signifie d’abord « appliquer le modèle ». Des scores de précision ne sont
calculés que si de vraies annotations sont fournies.

## Commencer

1. Suivre [l’installation](docs/INSTALLATION.md) une fois.
2. Préparer les [données](docs/DONNEES.md) et les [modèles](docs/MODELES.md).
3. Ouvrir **`configs/config.yaml`**, choisir `mode: train` ou `mode: test`,
   puis renseigner les chemins de votre dataset et du modèle.
4. Dans un terminal ouvert dans ce dossier :

```bash
python run.py --check
python run.py
```

Le [guide débutant pas à pas](docs/DEMARRAGE.md) explique les deux parcours.
On peut également remplacer temporairement le choix du fichier :

```bash
python run.py --mode train
python run.py --mode test
```

Changer de mode ne nécessite aucune modification du code Python. Les deux blocs
`training:` et `test:` restent dans le même fichier, prêts à être réutilisés.

**Les données et les poids des modèles ne sont pas inclus dans GitHub.** Le dépôt ne peut pas
produire des prédictions sans le couple configuration + poids. Le modèle BFMS est référencé sur Figshare et les modèles publics sur MMSegmentation.
Le modèle fine-tuné de l’autrice et les données sont distribués séparément sur Drive
(lien Drive à compléter). Voir [les sources des modèles](docs/MODELES.md) et
[le rôle de chaque dossier Drive](docs/DRIVE.md).

## Ce qui peut être activé en mode test

Par défaut, seule la segmentation sémantique est lancée, avec des aperçus.
Dans le bloc `test:` du YAML :

| Paramètre | Effet | Conditions |
|---|---|---|
| `labels_dir: null` | Pas de score, seulement des prédictions | Pour les scores, remplacer `null` par le dossier de masques annotés |
| `materials: true` | Matériaux BFMS, fusion et albédo | Modèle BFMS local complet |
| `indicators: true` | SVF, GVI, bâti et perméabilité | `materials: true`, panoramas 360° complets 2:1, horizon centré |
| `analysis: true` | Analyse des relations avec le microclimat | `indicators: true`, capteurs et horaires de votre acquisition |

Les chiffres SVF/GVI décrivent la vue depuis la caméra. La perméabilité et
l’albédo sont des estimations fondées sur des classes et des hypothèses ; ce ne
sont pas des mesures physiques directes. Lire [les limites](docs/METHODOLOGIE.md).

## Résultats

Chaque lancement crée `outputs/train_<date>/` ou `outputs/test_<date>/`.
La configuration utilisée, les versions de bibliothèques et le statut sont
sauvegardés. Les résultats précédents ne sont pas écrasés.

Après `train`, le terminal donne les deux chemins à reporter dans `test` :
`model_config` et `checkpoint`. Les [sorties expliquées](docs/RESULTATS.md)
indiquent quels fichiers ouvrir et comment comprendre les scores.

## Organisation du dépôt

| Dossier / fichier | Rôle |
|---|---|
| `run.py` | Lanceur simple |
| `configs/config.yaml` | Configuration principale train/test |
| `src/coolpath/cli.py` | Contrôles et enchaînement des étapes |
| `src/coolpath/data/` | Images, vidéo, annotations CVAT, découpage du dataset |
| `src/coolpath/segmentation/` | Entraînement, export et inférence |
| `src/coolpath/materials/` | BFMS, fusion et albédo |
| `src/coolpath/indicators/` | Géométrie sphérique et indicateurs urbains |
| `src/coolpath/thermal/`, `analysis/` | Capteurs et statistiques |
| `docs/` | Guides d’utilisation, méthodologie et dépannage |
| `scripts/` | Étapes individuelles pour usage avancé |
| `configs/pipeline.yaml` | Ancien exemple expérimental, utilisé seulement par les scripts numérotés |
| `tests/` | Vérifications numériques et du fonctionnement train/test |

Les notebooks 22, 23, 24, 25, 29 et 30 sont les références. La correspondance et
les changements sont décrits dans [MIGRATION.md](MIGRATION.md). Les notebooks
avec leurs images intégrées et l’historique de conversation ne sont pas recopiés
dans ce dépôt Python.

## Validation et publication

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Le détail des vérifications effectuées et de leurs limites est dans
[VALIDATION.md](VALIDATION.md). Le guide [Publier sur GitHub](docs/GITHUB.md)
explique comment déposer le projet et rendre les modèles disponibles aux autres.
