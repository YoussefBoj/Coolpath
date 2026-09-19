"""Validate indexed masks and produce reproducible, non-overlapping splits."""
from __future__ import annotations
import csv
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def inspect_images(folder, panorama=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Dossier images introuvable : {folder}')
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(
            f'Aucune image JPG/PNG dans {folder} (sous-dossiers non parcourus).'
        )
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError(
            'Noms ambigus : chaque image doit avoir un nom unique, sans son extension.'
        )
    for p in paths:
        with Image.open(p) as im:
            im.load()
            if panorama and abs(im.width / im.height - 2) >= .05:
                raise ValueError(
                    f'{p.name} : les indicateurs exigent des panoramas 360° complets au ratio 2:1.'
                )
    return paths


def inspect_labels(images, labels_dir):
    labels_dir = Path(labels_dir)
    for p in images:
        label = labels_dir / f'{p.stem}.png'
        if not label.is_file():
            raise ValueError(f'Annotation absente : {label}')
        with Image.open(label) as im:
            mask = np.asarray(im)
        with Image.open(p) as im:
            shape = (im.height, im.width)
        if mask.ndim != 2 or mask.shape != shape:
            raise ValueError(
                f'{label.name} : masque attendu à un canal, de même taille que l’image.'
            )
        bad = np.setdiff1d(np.unique(mask), list(range(19)) + [255])
        if bad.size:
            raise ValueError(
                f'{label.name} : indices invalides {bad.tolist()}, attendus 0..18 ou 255.'
            )
        if not np.any(mask != 255):
            raise ValueError(
                f'{label.name} : tous les pixels sont ignorés (255).')


def make_splits(stems,
                fraction=.2,
                seed=42,
                strategy='random',
                groups=None,
                split_dir=None):
    stems = sorted(stems)
    if not 0 < fraction < 1 or len(stems) < 2:
        raise ValueError(
            'Il faut au moins deux images et 0 < val_fraction < 1.')
    rng = random.Random(seed)
    if strategy == 'files':
        if split_dir is None:
            raise ValueError(
                'training.split_dir est requis avec split_strategy: files.')

        def read(name):
            return [
                s.strip()
                for s in (Path(split_dir) /
                          name).read_text(encoding='utf-8').splitlines()
                if s.strip()
            ]

        train, val = read('train.txt'), read('val.txt')
    elif strategy == 'group':
        if not groups or set(groups) != set(stems):
            raise ValueError(
                'groups.csv doit associer chaque image à un groupe (colonnes image,group).'
            )
        names = sorted(set(groups.values()))
        if len(names) < 2:
            raise ValueError(
                'Le découpage par groupes exige au moins deux vidéos/parcours distincts.'
            )
        rng.shuffle(names)
        n = min(len(names) - 1, max(1, round(len(names) * fraction)))
        val_groups = set(names[:n])
        val = [s for s in stems if groups[s] in val_groups]
        train = [s for s in stems if groups[s] not in val_groups]
    elif strategy == 'random':
        shuffled = stems.copy()
        rng.shuffle(shuffled)
        n = min(len(stems) - 1, max(1, round(len(stems) * fraction)))
        val, train = shuffled[:n], shuffled[n:]
    else:
        raise ValueError('split_strategy doit être random, group ou files.')
    if not train or not val or set(train) & set(val):
        raise ValueError(
            'Train et validation doivent être non vides et disjoints.')
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError('Les listes de découpage contiennent des doublons.')
    if set(train) | set(val) != set(stems):
        raise ValueError(
            'Les splits doivent couvrir exactement les images, sans extension ni chemin.'
        )
    return sorted(train), sorted(val)


def read_groups(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(not r.get('image') or not r.get('group') for r in rows):
        raise ValueError(
            'groups.csv : colonnes image,group attendues ; noms sans extension.'
        )
    result = {r['image']: r['group'] for r in rows}
    if len(result) != len(rows):
        raise ValueError('groups.csv contient des images en double.')
    return result


def prepare_indexed_dataset(images, labels_dir, destination, train, val):
    """Normalize mixed JPG/PNG input to RGB PNG, without JPEG recompression."""
    destination = Path(destination)
    for sub in ['images', 'labels', 'splits']:
        (destination / sub).mkdir(parents=True, exist_ok=True)
    for p in images:
        with Image.open(p) as im:
            im.convert('RGB').save(destination / 'images' / f'{p.stem}.png')
        # Palette indices (mode P) remain class indices, not RGB colors.
        with Image.open(Path(labels_dir) / f'{p.stem}.png') as im:
            Image.fromarray(np.asarray(im).astype(np.uint8)).save(
                destination / 'labels' / f'{p.stem}.png')
    for name, items in [('train', train), ('val', val)]:
        (destination / 'splits' / f'{name}.txt').write_text('\n'.join(items) +
                                                            '\n',
                                                            encoding='utf-8')
    return destination
