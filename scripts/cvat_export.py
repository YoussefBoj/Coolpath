"""Create labeled previews of colored CVAT masks (not a CVAT export)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def create_previews(images_dir, masks_dir, labelmap, output_dir, alpha=.55, min_area_fraction=.004):
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.measure import label, regionprops

    if not 0 <= alpha <= 1 or not 0 <= min_area_fraction <= 1:
        raise ValueError('alpha et min-area-fraction doivent être compris entre 0 et 1.')
    images_dir, masks_dir, labelmap, output_dir = map(Path, (images_dir, masks_dir, labelmap, output_dir))
    palette = []
    for line in labelmap.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split(':')
        color = tuple(int(v) for v in fields[1].split(','))
        if len(color) != 3 or any(v < 0 or v > 255 for v in color):
            raise ValueError(f'Couleur invalide : {line}')
        palette.append((fields[0], color))
    if not palette:
        raise ValueError('labelmap.txt ne contient aucune classe.')
    images = {}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if p.stem in images:
                raise ValueError(f'Nom d’image ambigu : {p.stem}')
            images[p.stem] = p
    masks = sorted(p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    if not masks:
        raise ValueError('Aucun masque PNG trouvé.')
    if output_dir.exists() and any(p.name != '.gitkeep' for p in output_dir.iterdir()):
        raise ValueError('Le dossier de sortie doit être vide pour ne rien écraser.')
    # Validate all pairs first, without resizing annotations silently.
    for mask_path in masks:
        if mask_path.stem not in images:
            raise ValueError(f'Image manquante pour {mask_path.name}')
        with Image.open(images[mask_path.stem]) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f'Dimensions différentes : {mask_path.name}')
            if mask.mode not in {'RGB', 'RGBA', 'P'}:
                raise ValueError(f'{mask_path.name} : masque CVAT coloré requis, pas un masque d’indices.')
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in masks:
        with Image.open(images[mask_path.stem]) as im:
            image = np.asarray(im.convert('RGB'))
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert('RGB'))
        fig, ax = plt.subplots(figsize=(14, 8))
        try:
            ax.imshow(((1 - alpha) * image + alpha * mask).astype(np.uint8))
            ax.axis('off')
            threshold = max(1, int(mask.shape[0] * mask.shape[1] * min_area_fraction))
            for name, color in palette:
                if name == 'background':
                    continue
                region = np.all(mask == np.asarray(color), axis=-1)
                for component in regionprops(label(region, connectivity=2)):
                    if component.area >= threshold:
                        y, x = component.centroid
                        ax.text(x, y, name, ha='center', va='center', color='white', fontsize=8,
                                bbox=dict(facecolor='black', alpha=.7, edgecolor='none'))
            ax.set_title(mask_path.stem)
            fig.savefig(output_dir / f'{mask_path.stem}_overlay.png', dpi=100, bbox_inches='tight')
        finally:
            plt.close(fig)
    return len(masks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['images-dir', 'masks-dir', 'labelmap', 'output-dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--alpha', type=float, default=.55)
    parser.add_argument('--min-area-fraction', type=float, default=.004)
    args = parser.parse_args(argv)
    try:
        count = create_previews(args.images_dir, args.masks_dir, args.labelmap, args.output_dir,
                                args.alpha, args.min_area_fraction)
    except Exception as exc:
        print(f'Erreur : {exc}', file=sys.stderr)
        return 2
    print(f'{count} aperçu(s) créé(s) dans {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
