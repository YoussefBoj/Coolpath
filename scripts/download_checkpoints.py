"""Download the public Cityscapes checkpoints identified in the original scripts."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'segformer': 'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024',
    'mask2former': 'mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024',
    'deeplabv3plus': 'deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024',
    'hrnet': 'fcn_hr48_4xb2-160k_cityscapes-512x1024',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['all', *MODELS], default=['mask2former'])
    parser.add_argument('--out-dir', default='checkpoints', help='Relative to the project root, or absolute')
    args = parser.parse_args(argv)
    executable = shutil.which('mim')
    if executable is None:
        print('MIM absent. Dans votre environnement actif : python -m pip install openmim', file=sys.stderr)
        return 2
    out = Path(args.out_dir).expanduser()
    out = out.resolve() if out.is_absolute() else (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = list(MODELS) if 'all' in args.models else list(dict.fromkeys(args.models))
    failed = []
    for name in selected:
        print(f'Téléchargement public {name} vers {out}', flush=True)
        cmd = [executable, 'download', 'mmsegmentation', '--config', MODELS[name], '--dest', str(out)]
        try:
            subprocess.run(cmd, check=True)  # No shell: portable paths, including spaces.
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f'Échec pour {name} : {exc}', file=sys.stderr)
            failed.append(name)
    print(f'{len(selected) - len(failed)}/{len(selected)} téléchargement(s) terminé(s).')
    if failed:
        print('Vérifier réseau, installation MMSegmentation et disponibilité des identifiants.', file=sys.stderr)
        return 1
    print('Reporter les vrais noms .py/.pth dans votre YAML. Ce ne sont pas les poids fine-tunés CoolPath.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
