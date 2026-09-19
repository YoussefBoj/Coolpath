# Installation

## 1. Choisir son environnement

Le projet s’appuie sur l’environnement OpenMMLab des notebooks. Utiliser
**Python 3.10** pour limiter les problèmes de roues binaires. Le mode `train`
implémenté ici demande un GPU NVIDIA avec CUDA. Le mode `test` accepte `cpu`,
mais Mask2Former reste coûteux. Les réglages du notebook visaient environ 16 Go
de mémoire GPU ; cela ne constitue pas une garantie pour toutes les résolutions.

Le profil ci-dessous est proposé pour Linux/WSL2 avec un GPU compatible CUDA
11.8. Il n’a pas été exécuté sur GPU lors de la préparation de ce ZIP. Pour une
carte récente incompatible avec cette génération de PyTorch/CUDA, adapter toute
la pile ensemble ; ne pas simplement installer la dernière version de MMCV.

## 2. Créer un environnement isolé

Sous Windows, utiliser Python 3.10 et l’invite de commandes :

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
```

Sous Linux / WSL2 :

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Les commandes suivantes se lancent dans le dossier du projet, environnement activé.

## 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour le profil NVIDIA CUDA 11.8 :

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 --only-binary=:all: -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -r requirements-models.txt
python -m pip install -e .
python -m pip check
```

`--only-binary` évite de déclencher involontairement une compilation longue de
MMCV. Si aucune roue n’est disponible pour votre système, suivre le guide
[MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
`mmcv-lite` ne fournit pas les opérations compilées nécessaires à Mask2Former.
Ne pas installer `mmcv` et `mmcv-lite` ensemble.

La documentation [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)
explique que MMCV dépend de la combinaison PyTorch/CUDA et que les poids et la
configuration sont nécessaires pour l’inférence. Pour CPU uniquement, choisir
PyTorch CPU et une construction MMCV CPU compatible selon ces guides, puis
régler `test.device: cpu`. Windows natif peut nécessiter une compilation ; WSL2
est une autre possibilité si aucune roue ne convient.

Pour les corrélations thermiques :

```bash
python -m pip install -r requirements-analysis.txt
```

## 4. Vérifier

```bash
python -c "import torch; from mmcv.ops import point_sample; import mmseg, mmdet; print('CUDA disponible :', torch.cuda.is_available())"
python run.py --help
```

Après avoir ajouté les données et les modèles :

```bash
python run.py --check-env
```

`pip install -e .` installe le paquet en mode développement et rend aussi
`coolpath` et `python -m coolpath` disponibles. Les dépendances GPU restent
volontairement séparées des dépendances de contrôle des données.

## Tests sans GPU

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
```

Ces tests ne téléchargent pas de poids et ne réalisent pas d’apprentissage.
