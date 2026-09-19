# Méthodologie et limites

## Ce qui vient des notebooks

Le projet sépare le jeu annoté qui sert à l’apprentissage du jeu complet traité
en inférence. Les nombres 116 images annotées et 1 041 images de parcours
caractérisent l’expérience d’origine, pas des tailles imposées aux utilisateurs.

Les modules reprennent l’extraction, la segmentation sémantique, BFMS, la fusion,
les indicateurs sphériques et les analyses thermiques du ZIP initial. Cette
version ajoute le lanceur train/test, des contrôles d’entrée et une documentation
pour d’autres datasets. Ce n’est pas une nouvelle validation scientifique de
chaque hypothèse des notebooks.

## Entraînement

Les classes restent les 19 classes Cityscapes. Le fine-tuning utilise des crops
512 × 1 024, un redimensionnement aléatoire, des flips horizontaux et une
augmentation photométrique. Les paramètres de nombre d’itérations, lot, mémoire
mixte et taux d’apprentissage sont configurables. Le seed est propagé à la
configuration, sans promettre une déterminisme bit à bit des opérations GPU.

Le choix du checkpoint est fondé sur la validation. Pour évaluer le résultat,
prévoir un troisième ensemble de test indépendant. Un découpage aléatoire de
frames quasi identiques peut surestimer les performances : les groupes par
parcours sont préférables. Les scripts historiques conservent la stratification
interne à chaque vidéo ; les deux procédures ne sont pas équivalentes.

## Matériaux et albédo

BFMS a montré des limites sur le domaine du projet. Les règles de fusion
remplacent les matériaux incompatibles avec la classe sémantique par une valeur
plausible. Une carte visuellement cohérente après correction ne prouve pas que
BFMS a reconnu correctement les matériaux. Le taux de correction est exporté.

La convention BFMS concernant le canal no-object est conservée du notebook 24.
Le chargement du processeur a été rendu explicite avec son fichier de
configuration réel. Les paramètres de normalisation doivent correspondre au
modèle utilisé ; ce choix doit être contrôlé lors de la première exécution GPU.

L’albédo repose sur une table par matériau, avec surcharges de contexte. Une
couleur moyenne de peinture, corrigée de l’ombre, est appariée à la table RAL.
Le code additionne les RGB corrigés par pixel sur tout le jeu de données, puis
calcule une moyenne globale sans conserver tous les pixels en mémoire.
L’albédo de peinture est constant pendant un lancement ; si deux datasets sont
traités séparément, leurs valeurs estimées peuvent différer. Pour les comparer
avec le même paramètre, fixer `fusion.paint_albedo` explicitement. En absence de
pixels de peinture, la valeur de repli 0,45 est enregistrée avec sa raison.

Les tables RAL et d’albédo viennent du notebook 25. Les couleurs d’une caméra ne
constituent pas une mesure radiométrique de réflectance. Les références et droits
de réutilisation des tables sont à conserver lors d’une publication scientifique.

## Indicateurs

SVF utilise une pondération sin(theta) cos(theta) sur l’hémisphère supérieur.
GVI et bâti conservent des fractions brutes de pixels et les variantes
pondérées par angle solide disponibles dans les modules.

La classe terrain inclut du sol nu : la compter avec vegetation ne permet pas
d’affirmer que tous les pixels comptés sont verts. Les seuils de séparation
buisson/canopée (-7° et 12°) sont des heuristiques à recalibrer visuellement selon
la caméra, sa hauteur et le lieu. La perméabilité qualifie le sol observable
selon les matériaux prédits, pas une capacité d’infiltration mesurée.

## Relations microclimatiques

Les fonctions statistiques héritées traitent l’inertie du globe, les fenêtres
historiques, les corrélations, l’autocorrélation et les tests multiples. Leur
interprétation exige une acquisition synchronisée et assez d’observations utiles.
Une corrélation ne démontre pas une causalité. Les tableaux vides, valeurs NaN
ou avertissements statistiques peuvent signaler des fenêtres insuffisantes ou
des variables constantes.

Le mode test principal conserve le format temporel GoPro/Comfy’Pack du notebook
30. Il ne prétend pas accepter automatiquement toutes les stations météo.
Les paramètres physiques des fonctions `thermal/comfort.py` et de l’analyse
restent ceux du projet ; les examiner avant d’utiliser un autre instrument.

## Portée de la livraison

Les sorties numériques sont conservées, mais les nombreuses figures de
présentation, cellules exploratoires et affichages interactifs des notebooks ne
sont pas tous reproduits. Les nouveaux aperçus servent au contrôle sémantique.
Aucune équivalence numérique complète avec les 1 041 images originales n’a pu
être mesurée sans ces images, les poids et les capteurs d’origine.
