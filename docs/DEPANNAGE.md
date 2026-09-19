# Dépannage

| Message / symptôme | Que faire ? |
|---|---|
| Aucune image | Vérifier le dossier du YAML, les extensions et l’absence de sous-dossiers |
| Modèle introuvable | Ajouter le couple `.py` / `.pth` et renseigner son chemin exact |
| Indices invalides | Utiliser les train IDs 0–18/255, pas les IDs Cityscapes bruts |
| Masque attendu à un canal | Fournir des PNG indexés ; pour les couleurs CVAT utiliser `dataset_format: cvat` |
| Deux images avec le même nom | Renommer les paires image/annotation de façon unique |
| Les indicateurs exigent 2:1 | Utiliser de vrais panoramas 360° complets, ou désactiver les indicateurs |
| CUDA indisponible | Vérifier le pilote/PyTorch, ou choisir CPU pour le mode test |
| CUDA out of memory en train | Réduire `batch_size` à 1 ; l’accumulation permet de conserver un lot effectif plus grand |
| CUDA out of memory avec BFMS | Réduire `bfms.scales` à `[768]`, ce qui change le protocole multiscale |
| No module named mmcv._ext | Installer MMCV avec opérations compilées, compatible avec PyTorch et CUDA |
| Erreur de version mmcv / mmdet | Reprendre ensemble les versions du guide d’installation dans un environnement propre |
| preprocessor_config.json absent | Obtenir le vrai processeur BFMS auprès de l’auteur du modèle |
| Masques presque noirs | Les fichiers contiennent des indices ; ouvrir les aperçus pour voir les couleurs |
| Résultats absents après arrêt | Vérifier `status.json`, `error.log` et les logs d’entraînement |
| Dossier de résultats existant | Mettre `run_name: null` ou choisir un nouveau nom |
| Pas de scores de segmentation | Fournir `test.labels_dir` avec de vraies annotations |
| Horodatage impossible | Vérifier `__f`, les noms de chapitres, `analysis.day`, `t0` et `fps` |

Utiliser `python run.py --debug` pour obtenir la trace technique complète. Pour
rapporter une erreur, joindre le message, le mode, les versions et un exemple
minimal de structure des données, sans publier des images privées ni des poids
non redistribuables. Le projet n’a pas besoin d’une clé API.
