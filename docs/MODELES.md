# Quels modèles fournir ?

Un fichier de poids ne suffit pas. Il faut aussi sa configuration, la bonne
architecture et le même ordre de classes. Le dépôt ne télécharge pas
silencieusement des modèles à votre place.

| Usage | Fichiers attendus | Où les indiquer |
|---|---|---|
| Test sémantique | Configuration MMSeg `.py` + checkpoint `.pth`, 19 classes Cityscapes | `test.model_config`, `test.checkpoint` |
| Fine-tuning | Configuration Mask2Former Swin-B Cityscapes + poids complets compatibles | `training.base_config`, `training.start_checkpoint` |
| Matériaux | `config.json`, `preprocessor_config.json`, `model.safetensors` (ou poids PyTorch) | `bfms.model_dir` |

## Utiliser les modèles CoolPath d’origine

Les notebooks référencent des fichiers locaux ; les `.pth`, les poids BFMS et
leur configuration complète ne sont pas joints au ZIP initial. Il faut donc les
récupérer auprès de la personne qui détient les entraînements d’origine. Aucune
URL publique de ces poids n’a été fournie.

Pour que les autres puissent utiliser immédiatement le projet, le propriétaire
du dépôt devra publier un lien de téléchargement vers :

1. Le meilleur checkpoint sémantique et sa configuration résolue.
2. Le dossier BFMS complet, si la segmentation matériaux est proposée.
3. Une indication claire des classes, de la version du modèle, des conditions
   d’utilisation et du jeu d’évaluation associé.

Ne pas publier uniquement le fichier `.py` généré d’un ancien entraînement s’il
hérite d’une configuration `_base_` restée sur un ordinateur personnel.

## Alternative : démarrer depuis un modèle Cityscapes public

Si vous n’avez pas les poids CoolPath, vous pouvez partir d’un Mask2Former Swin-B
Cityscapes de la [bibliothèque officielle MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2/configs/mask2former),
puis faire votre propre fine-tuning. Ce modèle générique n’est pas le modèle
CoolPath déjà adapté et ne reproduit pas ses performances.

Après installation d’OpenMMLab, exemple de récupération avec MIM :

```bash
python -m pip install openmim
mim download mmsegmentation --config mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024 --dest models/pretrained
```

Le checkpoint téléchargé a un nom long. Renseigner dans le YAML les **noms réels**
des fichiers obtenus. Si une configuration contient `_base_`, conserver tous les
fichiers référencés ou la résoudre avec `mmengine.Config.fromfile(...).dump(...)`.
Un modèle ADE20K à 150 classes ou un simple poids de backbone Swin ne remplace
pas un checkpoint complet Cityscapes à 19 classes.

Le mode `train` exporte une configuration d’inférence résolue et le checkpoint
sélectionné dans `model/`. Après export, ces deux fichiers peuvent être déplacés
ensemble vers une autre machine disposant des mêmes bibliothèques.

## BFMS

Le modèle matériaux doit respecter les **43 classes et indices** de
`src/coolpath/taxonomy.py`. Le contrôle refuse un autre ordre ; il ne faut pas
renommer arbitrairement les classes pour le contourner.

Le processeur est chargé depuis `preprocessor_config.json`, avec ses paramètres
réels de normalisation, redimensionnement et padding. Un `config.json` de réseau
ne remplace pas automatiquement la configuration du processeur. Si le dossier
historique ne possède pas ce fichier, exporter le processeur utilisé lors de
l’entraînement avec `processor.save_pretrained(dossier)` dans son environnement
d’origine ; ne pas inventer ses paramètres.

Le calcul multi-échelle conserve la convention du notebook 24 : le canal
« no-object » participe au calcul puis devient Background. Cette convention est
explicitée car elle n’est pas interchangeable avec tous les post-traitements
standards. Les APIs du modèle/processeur sont documentées dans
[Transformers Mask2Former](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/mask2former).
