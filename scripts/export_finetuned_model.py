"""Export a historical MMSeg config and its chosen weights for inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys


def remove_initialization(value):
    """A full inference checkpoint replaces external model initializers."""
    if isinstance(value, dict):
        return {key: None if key in {'init_cfg', 'pretrained'} else remove_initialization(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [remove_initialization(item) for item in value]
    if isinstance(value, tuple):
        return tuple(remove_initialization(item) for item in value)
    return value


def export(config_path, checkpoint, output):
    from mmengine.config import Config
    config_path, checkpoint, output = map(lambda p: Path(p).expanduser().resolve(),
                                           (config_path, checkpoint, output))
    if not config_path.is_file() or not checkpoint.is_file():
        raise ValueError('La configuration et le checkpoint sélectionné doivent exister.')
    if output.exists() and any(p.name != '.gitkeep' for p in output.iterdir()):
        raise ValueError('Destination non vide : choisir un nouveau dossier pour ne rien écraser.')
    config = Config.fromfile(str(config_path))
    if 'model' not in config or 'test_pipeline' not in config:
        raise ValueError('La configuration résolue doit contenir model et test_pipeline.')
    head = config.model.get('decode_head', {})
    if not isinstance(head, dict) or head.get('num_classes') != 19:
        raise ValueError('Cet export CoolPath attend une tête sémantique à 19 classes.')
    # Inference needs no ground-truth annotation transform.
    pipeline = [item for item in config.test_pipeline if item.get('type') != 'LoadAnnotations']
    portable = dict(model=remove_initialization(config.model.to_dict()), test_pipeline=pipeline,
                    default_scope=config.get('default_scope', 'mmseg'))
    if config.get('custom_imports'):
        portable['custom_imports'] = config.custom_imports
    output.mkdir(parents=True, exist_ok=True)
    Config(portable).dump(str(output / 'model_config.py'))
    shutil.copy2(checkpoint, output / 'model.pth')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='The checkpoint you selected, not last_checkpoint.txt')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        result = export(args.config, args.checkpoint, args.output)
    except Exception as exc:
        print(f'Export impossible : {exc}\nRésoudre les fichiers _base_ sur la machine source et vérifier MMEngine.', file=sys.stderr)
        return 2
    print(f'Export créé : {result}\nPartager model_config.py et model.pth ensemble. Poids copiés, réseau non exécuté.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
