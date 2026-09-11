from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_EXTENSIONS)

def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Image illisible: {path}')
    return image

def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Masque illisible: {path}')
    return mask
