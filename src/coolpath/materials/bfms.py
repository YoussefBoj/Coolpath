from __future__ import annotations
from pathlib import Path
import time
import cv2, numpy as np, pandas as pd
from PIL import Image
from coolpath.taxonomy import BFMS_CLASSES
from coolpath.utils.io import ensure_dir, list_images

TARGET_GROUPS = {
    'vitrage_%': [15, 20, 27, 30],
    'asphalt_%': [9],
    'beton_%': [10],
    'dalles_pierre_%': [2, 40, 42]
}


def load_bfms(model_dir: str | Path, device='cuda'):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, AutoImageProcessor
    import torch
    model_dir = Path(model_dir)
    cfg = Mask2FormerConfig.from_pretrained(model_dir / 'config.json')
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir, config=cfg, local_files_only=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_dir,
                                                   local_files_only=True,
                                                   use_fast=False)
    if cfg.num_labels != 43:
        raise ValueError(
            'BFMS doit utiliser les 43 classes du projet, dans le même ordre.')
    return model, processor, device


def predict_bfms(model,
                 processor,
                 device,
                 img_bgr: np.ndarray,
                 scales=(768, 896, 1024)) -> np.ndarray:
    import torch
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w = image.height, image.width
    probs = []
    for s in scales:
        inputs = processor(images=image,
                           size={
                               'shortest_edge': s,
                               'longest_edge': int(s * 2)
                           },
                           return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        masks = torch.sigmoid(outputs.masks_queries_logits[0])
        classes = torch.softmax(outputs.class_queries_logits[0], dim=-1)
        pixel = torch.einsum('qc,qhw->chw', classes, masks)
        pixel = torch.nn.functional.interpolate(pixel.unsqueeze(0),
                                                size=(h, w),
                                                mode='bilinear',
                                                align_corners=False)[0]
        probs.append(pixel.cpu())
    mask = torch.argmax(torch.stack(probs).mean(0),
                        dim=0).numpy().astype(np.uint8)
    mask[mask >= 43] = 0
    return mask


def run_bfms(images_dir,
             output_dir,
             model_dir,
             device='cuda',
             scales=(768, 896, 1024),
             overwrite=False):
    import torch
    output_dir = Path(output_dir)
    masks_dir = ensure_dir(output_dir / 'masks_bfms')
    model, processor, device = load_bfms(model_dir, device)
    rows = []
    global_counts = np.zeros(43, dtype=np.int64)
    for i, p in enumerate(list_images(images_dir), 1):
        dst = masks_dir / f'{p.stem}_bfms.png'
        img = cv2.imread(str(p))
        if dst.exists() and not overwrite:
            mask = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
        else:
            mask = predict_bfms(model, processor, device, img, scales)
            cv2.imwrite(str(dst), mask)
        counts = np.bincount(mask.ravel(), minlength=43)[:43]
        global_counts += counts
        total = max(int(counts.sum()), 1)
        row = {'image': p.stem, 'pixels': total}
        for name, ids in TARGET_GROUPS.items():
            row[name] = round(float(counts[ids].sum() / total * 100), 2)
        rows.append(row)
        if i % 10 == 0: print(f'[{i}] {p.name}')
        if i % 5 == 0 and torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(output_dir / 'bfms_par_image.csv', index=False)
    dist = pd.DataFrame({
        'materiau':
        BFMS_CLASSES,
        'pixels':
        global_counts,
        'proportion_%':
        global_counts / max(global_counts.sum(), 1) * 100
    })
    dist[dist.pixels > 0].sort_values('pixels', ascending=False).to_csv(
        output_dir / 'dist_bfms_globale.csv', index=False)
