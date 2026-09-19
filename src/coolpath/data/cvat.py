from __future__ import annotations
from pathlib import Path
import random, shutil
import numpy as np
from PIL import Image
from coolpath.taxonomy import CITYSCAPES_NAME_TO_ID, IGNORE_INDEX
from coolpath.utils.io import ensure_dir


def parse_labelmap(path: str | Path) -> dict[tuple[int, int, int], str]:
    mapping = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split(':')
        mapping[tuple(int(v) for v in parts[1].split(','))] = parts[0].strip()
    return mapping


def convert_cvat_export(cvat_dir: str | Path, frames_source: str | Path,
                        dataset_dir: str | Path) -> dict:
    cvat_dir, frames_source, dataset_dir = map(
        Path, (cvat_dir, frames_source, dataset_dir))
    img_out, lbl_out = ensure_dir(dataset_dir / 'images'), ensure_dir(
        dataset_dir / 'labels')
    image_source = cvat_dir / 'JPEGImages' if (
        cvat_dir / 'JPEGImages').exists() else frames_source
    mapping = parse_labelmap(cvat_dir / 'labelmap.txt')
    mapping = {
        color: name.replace(' ', '_')
        for color, name in mapping.items()
    }
    unknown = set(
        mapping.values()) - set(CITYSCAPES_NAME_TO_ID) - {'background'}
    if unknown: raise ValueError(f'Classes CVAT inconnues : {sorted(unknown)}')
    if not list((cvat_dir / 'SegmentationClass').glob('*.png')):
        raise ValueError('Aucun masque CVAT dans SegmentationClass/.')
    if any(img_out.iterdir()) or any(lbl_out.iterdir()):
        raise ValueError(
            'Le dossier de conversion doit être vide pour éviter de mélanger les datasets.'
        )
    counts = np.zeros(19, dtype=np.int64)
    ignore = 0
    converted = 0
    for mp in sorted((cvat_dir / 'SegmentationClass').glob('*.png')):
        rgb = np.asarray(Image.open(mp).convert('RGB'))
        idx = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for color, name in mapping.items():
            sel = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
            if name != 'background' and name in CITYSCAPES_NAME_TO_ID:
                idx[sel] = CITYSCAPES_NAME_TO_ID[name]
        vals, cnts = np.unique(idx, return_counts=True)
        for v, c in zip(vals, cnts):
            if v == IGNORE_INDEX: ignore += int(c)
            else: counts[int(v)] += int(c)
        Image.fromarray(idx).save(lbl_out / f'{mp.stem}.png')
        src = next((image_source / f'{mp.stem}{ext}'
                    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
                    if (image_source / f'{mp.stem}{ext}').exists()), None)
        if src is None:
            raise FileNotFoundError(f'Image source introuvable: {mp.stem}')
        shutil.copy2(src, img_out / f'{mp.stem}{src.suffix.lower()}')
        converted += 1
    total = int(counts.sum() + ignore)
    return {
        'converted': converted,
        'class_pixel_counts': counts.tolist(),
        'ignore_fraction': ignore / max(total, 1)
    }


def grouped_train_val_split(images_dir: str | Path,
                            dataset_dir: str | Path,
                            val_fraction: float = .2,
                            seed: int = 42) -> tuple[list[str], list[str]]:
    images_dir, dataset_dir = Path(images_dir), Path(dataset_dir)
    # Legacy notebook-style stratification within each video, NOT held-out groups.
    stems = sorted(p.stem for p in images_dir.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not 0 < val_fraction < 1 or len(stems) < 2:
        raise ValueError('Au moins 2 images et 0 < val_fraction < 1 requis.')
    groups = {}
    for stem in stems:
        groups.setdefault(stem.split('_')[0], []).append(stem)
    rng = random.Random(seed)
    train, val = [], []
    for items in groups.values():
        rng.shuffle(items)
        n_val = min(len(items) - 1, max(1, round(val_fraction * len(items))))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    if not val:
        rng.shuffle(train)
        val.append(train.pop())
    split_dir = ensure_dir(dataset_dir / 'splits')
    (split_dir / 'train.txt').write_text('\n'.join(sorted(train)),
                                         encoding='utf-8')
    (split_dir / 'val.txt').write_text('\n'.join(sorted(val)),
                                       encoding='utf-8')
    return sorted(train), sorted(val)
