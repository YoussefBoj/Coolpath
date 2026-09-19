# Guide pas à pas

## Les mots utiles

- **Dataset** : votre collection d’images.
- **Annotation / masque de référence** : une image dont chaque pixel indique la
  catégorie correcte, dessinée ou vérifiée par une personne.
- **Poids / checkpoint** : fichier `.pth` contenant ce qu’un modèle a appris.
- **Configuration du modèle** : fichier `.py` qui décrit son architecture.
- **Inférence** : demander au modèle de prédire les catégories d’une image.
- **Validation** : mesurer l’apprentissage sur des images qui ne servent pas à
  mettre à jour les poids. Ces images servent cependant à choisir le modèle.

Décompresser le ZIP. Ouvrir le dossier `coolpath-pipeline` dans VS Code, puis
**Terminal → Nouveau terminal**. Toutes les commandes ci-dessous partent de ce
dossier. Installer les bibliothèques selon [INSTALLATION.md](INSTALLATION.md).

## A. Tester ses images avec un modèle existant

1. Mettre les images `.jpg`, `.jpeg` ou `.png` dans `data/test/images/`.
2. Mettre les poids sémantiques dans `models/semantic/model.pth` et leur
   configuration dans `models/semantic/model_config.py`. Ces noms sont des noms
   locaux de rangement, pas des fichiers fournis dans le ZIP.
3. Ouvrir `configs/config.yaml`. Conserver :

```yaml
mode: test
```

4. Dans le bloc `test:`, vérifier les chemins. Pour un premier essai, garder
   `materials`, `indicators` et `analysis` à `false`. Sans GPU, choisir
   `device: cpu` ; l’installation doit tout de même disposer des opérations MMCV.
5. Lancer :

```bash
python run.py --check
python run.py --check-env
python run.py
```

`--check` lit et contrôle les fichiers, sans charger les poids. `--check-env`
vérifie aussi les bibliothèques et la disponibilité du périphérique. Le
lancement réel est nécessaire pour vérifier la compatibilité architecture/poids.

6. Ouvrir `outputs/test_<date>/semantic/previews/` pour voir les images colorées.
   `semantic/masks/` contient les indices de classes et peut sembler très sombre
   dans une visionneuse : c’est normal.

Avec des annotations de test, renseigner `test.labels_dir`. Les scores seront
alors dans `evaluation/`. Utiliser un jeu indépendant de l’entraînement et de
la validation pour estimer la généralisation.

## B. Entraîner sur ses propres images

**Des images seules ne suffisent pas : il faut leurs annotations.**

1. Mettre les images dans `data/train/images/` et les masques indexés dans
   `data/train/labels/`. Une paire doit porter le même nom sans extension.
   Pour un export CVAT, suivre la variante de [DONNEES.md](DONNEES.md).
2. Ajouter une configuration Mask2Former Swin-B Cityscapes et les poids de
   départ compatibles dans `models/pretrained/` (voir [MODELES.md](MODELES.md)).
3. Modifier uniquement le choix général :

```yaml
mode: train
```

4. Vérifier le bloc `training:`. Les valeurs initiales reprennent le projet :
   5 000 itérations, validation toutes les 250 itérations, lot de 2 images.
   Avec des frames proches, fournir `groups.csv` et choisir `split_strategy: group`.
5. Exécuter `python run.py --check`, puis `python run.py`.
6. Le modèle exporté se trouve dans `outputs/train_<date>/model/`. Les fichiers
   de découpage train/validation sont conservés dans `dataset/splits/`.

Ce mode adapte le modèle initial ; il ne crée pas une nouvelle architecture et
ne réentraîne pas BFMS. Les 19 catégories restent celles du projet.

## C. Tester le modèle que l’on vient d’entraîner

Dans **le même** `configs/config.yaml`, mettre `mode: test`, puis reporter dans
`test:` les chemins affichés à la fin de l’entraînement, par exemple :

```yaml
test:
  images_dir: data/test/images
  model_config: outputs/train_20260918_120000_000000/model/model_config.py
  checkpoint: outputs/train_20260918_120000_000000/model/model.pth
```

Ce fragment illustre trois paramètres : **ne pas remplacer tout le bloc test** ;
garder les autres paramètres existants. Utiliser le véritable nom du dossier
créé chez vous. Exécuter `python run.py`.

## D. Activer le projet complet

Ajouter le modèle BFMS, mettre `materials: true` et `indicators: true`.
Les images doivent être des panoramas équirectangulaires complets au ratio 2:1,
avec horizon au milieu de l’image, sans rotation verticale. Le ratio seul ne
prouve pas que l’image est correctement orientée.

Pour ajouter les corrélations, installer `requirements-analysis.txt`, mettre
`analysis: true` et renseigner le CSV capteurs, la date, les débuts de chapitres
et les FPS originaux. Ne jamais réutiliser les horaires d’une autre acquisition.

## Arrêter ou reprendre un entraînement

`Ctrl+C` arrête le processus. Si un checkpoint complet a déjà été sauvegardé,
mettre son chemin dans `training.resume_from`, garder les mêmes données, splits
et paramètres d’optimisation, puis relancer. Un nouveau dossier sera créé.
`max_iters` est le nombre total visé, pas un nombre à ajouter. Le fichier exporté
`model/model.pth` sert surtout à l’inférence ; pour reprendre, préférer
`training/iter_<nombre>.pth`, qui contient l’état d’entraînement.

La reprise explicite utilise les options décrites dans la
[documentation MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html).
