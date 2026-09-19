# Modèles publics et modèle fine-tuné CoolPath

## Trois ressources différentes

1. **Modèles sémantiques publics** : checkpoints Cityscapes de MMSegmentation,
   utilisés comme références de benchmark ou point de départ d’apprentissage.
2. **Modèle sémantique fine-tuné par l’autrice** : Mask2Former adapté au projet,
   conservé avec les résultats de `final_training_m2f` sur le Drive annoncé.
3. **BFMS** : modèle préentraîné de segmentation des matériaux, distribué via Figshare.

Un checkpoint public Cityscapes ne contient pas le fine-tuning personnel de
l’autrice. Pour utiliser celui-ci, télécharger ses poids et sa configuration
portable depuis le Drive, selon [DRIVE.md](DRIVE.md).

## Modèles sémantiques utilisés

Les identifiants ci-dessous viennent des scripts de téléchargement fournis par
l’autrice. Les liens sont ceux qu’elle a communiqués. Leur contenu distant n’a
pas pu être vérifié depuis l’environnement de préparation de cette mise à jour.

| Modèle | Configuration demandée par les scripts | Source |
|---|---|---|
| SegFormer MiT-B5 | `segformer_mit-b5_8xb1-160k_cityscapes-1024x1024` | [SegFormer](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer) |
| Mask2Former Swin-B | `mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024` | [Mask2Former](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/mask2former) |
| DeepLabV3+ ResNet-101 | `deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024` | [DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus) |
| HRNet-W48 avec tête FCN | `fcn_hr48_4xb2-160k_cityscapes-512x1024` | [HRNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/hrnet) |

Les pages GitHub contiennent les configurations et les références des modèles.
Ce ne sont pas quatre liens directs vers des fichiers `.pth`. Les liens `main`
peuvent évoluer ; le dépôt Python vise MMSegmentation 1.2.2. MIM utilise le
catalogue de la version installée. Conserver ensemble configuration et poids
provenant du même modèle.

Après [installation](INSTALLATION.md), installer MIM et télécharger uniquement
ce dont on a besoin :

```bash
python -m pip install openmim
python scripts/download_checkpoints.py --models mask2former
```

Ou récupérer les quatre références du benchmark :

```bash
python scripts/download_checkpoints.py --models all
```

HRNet seul, avec l’ancien nom de script conservé :

```bash
python scripts/download_HRnet.py
```

Les téléchargements vont dans `checkpoints/`, relativement à la racine du dépôt,
même si le script est appelé depuis un autre dossier. `--out-dir` permet de
changer la destination. Aucun téléchargement ne part simplement en important
les scripts. Les échecs donnent un code de sortie non nul.

Renseigner ensuite les **noms réellement téléchargés** dans
`training.base_config` et `training.start_checkpoint`. Le script ne renomme pas
arbitrairement les poids et ne modifie pas votre YAML.

Le notebook 23 repartait de préférence d’un checkpoint déjà fine-tuné du
notebook 18, avec repli sur Cityscapes. Repartir des poids publics permet un
nouvel apprentissage, mais ne reproduit pas exactement cette initialisation
historique sans le checkpoint intermédiaire.

Le mode `train` reste conçu pour **Mask2Former Swin-B**. Ajouter les quatre liens
ne transforme pas l’entraînement en benchmark automatique à quatre architectures.
Les autres modèles sont des références du travail initial.

## Utiliser le modèle fine-tuné de l’autrice

Voir [la description du Drive](DRIVE.md). Le dossier important est
`final_training_m2f`. Télécharger un checkpoint réellement sélectionné ainsi que
sa configuration d’inférence résolue. Le couple de distribution recommandé est :

| Fichier local | Rôle |
|---|---|
| `models/semantic/model.pth` | Poids fine-tunés |
| `models/semantic/model_config.py` | Architecture et prétraitement d’inférence |

Le lanceur accepte ces chemins dans `test.checkpoint` et `test.model_config`.
Il n’a pas besoin des logs, des courbes, de toutes les annotations ni de tous les
checkpoints pour traiter de nouvelles images.

Un `.py` historique qui dépend d’un `_base_` sur l’ordinateur de l’autrice n’est
pas encore portable. Utiliser `scripts/export_finetuned_model.py` sur la machine
source. Le mode `train` du nouveau dépôt produit déjà un export autonome.

## BFMS : matériaux

**Source communiquée par l’autrice : [BFMS sur Figshare](https://figshare.com/s/fd38d547fdb8708381f5).**
L’accès à cette page a renvoyé HTTP 403 lors de la préparation : les noms exacts
et le contenu de son archive ne sont pas confirmés ici.

Télécharger et extraire le modèle, puis indiquer son véritable dossier dans
`bfms.model_dir` (par défaut `models/bfms`). Le chargeur du dépôt attend :

| Fichier | Rôle |
|---|---|
| `config.json` | Architecture et correspondance des 43 classes matériaux |
| `model.safetensors` ou `pytorch_model.bin` | Paramètres appris ; les formats fragmentés compatibles utilisent aussi un fichier d’index |
| `preprocessor_config.json` | Paramètres réels du processeur d’images |

Ce tableau décrit le **format attendu par le code**, pas une inspection de
l’archive Figshare. Si le modèle est dans un sous-dossier de l’archive, pointer
`bfms.model_dir` sur celui qui contient ces fichiers.

L’ordre des 43 classes doit correspondre à `src/coolpath/taxonomy.py`. Ne pas
renommer arbitrairement les classes pour contourner le contrôle. Si la
configuration du processeur manque, récupérer le processeur de l’environnement
original et l’exporter avec `processor.save_pretrained(dossier)` ; ne pas
inventer les valeurs de normalisation.

Le dossier Drive `bfms_materials_1041` contient les résultats du traitement
historique ; il ne remplace pas le dossier de poids BFMS.
Le mode `test` active BFMS avec `materials: true`. Le mode `train` n’entraîne pas
BFMS : il adapte uniquement la segmentation sémantique.

La convention multi-échelle du notebook 24 reste conservée : le canal no-object
est ramené à Background. Les limites de BFMS et des règles de fusion restent
décrites dans [METHODOLOGIE.md](METHODOLOGIE.md).
