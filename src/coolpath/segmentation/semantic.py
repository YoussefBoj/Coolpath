from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from coolpath.utils.io import ensure_dir, list_images


def load_mmseg_model(config_path: str | Path,
                     checkpoint_path: str | Path,
                     device='cuda:0'):
    from mmseg.apis import init_model
    return init_model(str(config_path), str(checkpoint_path), device=device)


def predict_semantic(model, image) -> np.ndarray:
    from mmseg.apis import inference_model
    result = inference_model(model, image)
    return result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)


def run_inference(model,
                  images_dir: str | Path,
                  output_dir: str | Path,
                  overwrite=False):
    out = ensure_dir(output_dir)
    rows = []
    for i, p in enumerate(list_images(images_dir), 1):
        dst = out / f'{p.stem}_sem.png'
        if dst.exists() and not overwrite:
            rows.append(dst)
            continue
        mask = predict_semantic(model, str(p))
        cv2.imwrite(str(dst), mask)
        rows.append(dst)
        if i % 25 == 0: print(f'[{i}] {p.name}')
    return rows
