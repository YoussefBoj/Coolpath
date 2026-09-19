# Vérifications de cette livraison

## Résultats observés

- **25 tests réussis** avec `python -m pytest -q`.
- Compilation syntaxique de `src/`, `scripts/` et `run.py` réussie.
- Aide de `python run.py --help` vérifiée.
- Résolution de configuration avec **MMEngine 0.10.5** vérifiée : les paramètres
  du dataset Cityscapes hérités sont remplacés, la configuration exportée ne
  dépend plus de `_base_` et ne contient pas les chemins de données d’entraînement.
- Liens locaux de la documentation vérifiés.

## Ce que couvrent les tests

Géométrie SVF/GVI, règles matériaux, FDR/corrélation, résolution des chemins,
formats d’images et masques, annotations CVAT, séparation des groupes, jeux
vides/doublons, normalisation des suffixes, scores avec faux positifs, sélection
numérique des checkpoints, options de reprise, non-écrasement des résultats.

Le parcours test est exécuté avec des **prédictions synthétiques** à la frontière
des réseaux : les fichiers de masques, aperçus, scores, fusion et indicateurs
sont réellement produits. Le parcours train est testé avec le processus
d’apprentissage remplacé : données, splits et choix des chemins exportés sont
vérifiés. Ces tests ne mesurent pas la précision d’un réseau.

## Ce qui reste à valider sur la machine cible

Aucun poids sémantique ou BFMS n’était joint aux sources. Il n’y a pas eu
d’entraînement GPU, d’inférence avec ces vrais poids, ni de reproduction des
résultats sur le parcours original ou des analyses capteurs complètes. La pile
CUDA proposée dans le guide n’a pas été installée et validée de bout en bout ici.
Le workflow GitHub Actions est fourni mais n’a pas encore tourné sur votre dépôt.

Avant d’annoncer un résultat scientifique, effectuer un essai réel avec les
modèles, leur processeur et les données cibles ; contrôler les prédictions et
conserver un jeu de test indépendant.

## Environnement des contrôles locaux

Les tests numériques ont été exécutés dans l’environnement disponible ci-dessous.
Il est différent du profil GPU Python 3.10 proposé dans le guide.

- Python : 3.12.14
- numpy : 2.3.5
- pandas : 2.2.3
- scipy : 1.17.0
- Pillow : 12.3.0
- PyYAML : 6.0.3
- pytest : 9.1.1
- mmengine : 0.10.5
- scikit-image : 0.26.0
- opencv-python : 5.0.0.93
