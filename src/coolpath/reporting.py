"""Readable previews and optional held-out segmentation metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from coolpath.taxonomy import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE, colorize_cityscapes


def write_previews(images, masks_dir, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for p in images:
        with Image.open(Path(masks_dir) / f'{p.stem}_sem.png') as im:
            color = Image.fromarray(colorize_cityscapes(np.asarray(im)))
        with Image.open(p) as im:
            rgb = im.convert('RGB')
            Image.blend(rgb, color,
                        .5).save(destination / f'{p.stem}_overlay.jpg',
                                 quality=90)
    pd.DataFrame({
        'indice': range(19),
        'classe': CITYSCAPES_CLASSES,
        'R': CITYSCAPES_PALETTE[:, 0],
        'G': CITYSCAPES_PALETTE[:, 1],
        'B': CITYSCAPES_PALETTE[:, 2]
    }).to_csv(destination / 'legende.csv', index=False)


def evaluate_masks(images, masks_dir, labels_dir, destination):
    conf = np.zeros((19, 19), dtype=np.int64)
    for p in images:
        gt = np.asarray(Image.open(Path(labels_dir) / f'{p.stem}.png')).astype(
            np.int64)
        pred = np.asarray(Image.open(Path(masks_dir) /
                                     f'{p.stem}_sem.png')).astype(np.int64)
        if pred.shape != gt.shape or not np.isin(pred, range(19)).all():
            raise ValueError(f'Prédiction invalide : {p.name}')
        valid = gt != 255
        conf += np.bincount(gt[valid] * 19 + pred[valid],
                            minlength=361).reshape(19, 19)
    diagonal = conf.diagonal().astype(float)
    union = conf.sum(0) + conf.sum(1) - diagonal
    iou = np.divide(diagonal, union, out=np.full(19, np.nan), where=union > 0)
    present = conf.sum(1) > 0
    # Standard mean includes false-positive-only classes. Also expose notebook convention.
    metrics = dict(mIoU_pct=float(np.nanmean(iou) * 100),
                   mIoU_GT_present_pct=float(np.nanmean(iou[present]) * 100),
                   pixel_accuracy_pct=float(diagonal.sum() / conf.sum() * 100),
                   images=len(images))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'metrics.json').write_text(json.dumps(metrics, indent=2),
                                              encoding='utf-8')
    pd.DataFrame({
        'classe': CITYSCAPES_CLASSES,
        'IoU_pct': iou * 100,
        'pixels_GT': conf.sum(1)
    }).to_csv(destination / 'scores_par_classe.csv', index=False)
    pd.DataFrame(conf, index=CITYSCAPES_CLASSES,
                 columns=CITYSCAPES_CLASSES).to_csv(destination /
                                                    'confusion.csv')
    return metrics
