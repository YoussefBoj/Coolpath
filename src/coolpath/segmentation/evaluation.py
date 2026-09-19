from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from coolpath.taxonomy import CITYSCAPES_CLASSES, IGNORE_INDEX
from coolpath.segmentation.semantic import load_mmseg_model, predict_semantic


def evaluate_checkpoint(config_path,
                        checkpoint_path,
                        images_dir,
                        labels_dir,
                        stems,
                        image_suffix='.jpg',
                        device='cuda:0'):
    model = load_mmseg_model(config_path, checkpoint_path, device)
    n = len(CITYSCAPES_CLASSES)
    conf = np.zeros((n, n), dtype=np.int64)
    for stem in stems:
        pred = predict_semantic(
            model,
            str(Path(images_dir) / f'{stem}{image_suffix}')).astype(np.int64)
        gt = np.asarray(Image.open(Path(labels_dir) / f'{stem}.png')).astype(
            np.int64)
        valid = gt != IGNORE_INDEX
        conf += np.bincount(gt[valid] * n + pred[valid],
                            minlength=n * n).reshape(n, n)
    inter = np.diag(conf).astype(float)
    union = conf.sum(1) + conf.sum(0) - inter
    gt_px = conf.sum(1)
    present = gt_px > 0
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan) * 100
    acc = np.where(present, inter / np.maximum(gt_px, 1), np.nan) * 100
    metrics = {
        'mIoU': float(np.nanmean(iou[present])),
        'mAcc': float(np.nanmean(acc[present])),
        'aAcc': float(inter.sum() / max(conf.sum(), 1) * 100)
    }
    return conf, iou, acc, metrics
