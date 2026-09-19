"""Two-mode application: validate inputs first, then train or predict."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import sys
import traceback
import yaml
from coolpath.config import get_path, load_config, repo_path
from coolpath.data.dataset import inspect_images, inspect_labels, make_splits, read_groups, prepare_indexed_dataset


def require_file(path, description):
    if path is None or not path.is_file():
        raise ValueError(
            f'{description} introuvable : {path}. Voir docs/MODELES.md et docs/DONNEES.md.'
        )


def positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} doit être un entier strictement positif.')


def training_inputs(cfg):
    tr = cfg['training']
    if tr['dataset_format'] == 'indexed':
        images = inspect_images(get_path(cfg, 'training.images_dir'))
        inspect_labels(images, get_path(cfg, 'training.labels_dir'))
    elif tr['dataset_format'] == 'cvat':
        import tempfile
        from coolpath.data.cvat import convert_cvat_export
        # Validate actual conversion too; malformed annotations fail before GPU loading.
        with tempfile.TemporaryDirectory(prefix='coolpath-check-') as tmp:
            convert_cvat_export(get_path(cfg, 'training.cvat_export_dir'),
                                get_path(cfg, 'training.images_dir'), tmp)
            images = inspect_images(Path(tmp) / 'images')
            inspect_labels(images, Path(tmp) / 'labels')
    else:
        raise ValueError('dataset_format doit être indexed ou cvat.')
    groups_path = repo_path(tr.get('groups_csv'), cfg)
    groups = read_groups(
        groups_path
    ) if tr['split_strategy'] == 'group' and groups_path else None
    train, val = make_splits([p.stem for p in images], tr['val_fraction'],
                             tr['seed'], tr['split_strategy'], groups,
                             repo_path(tr.get('split_dir'), cfg))
    return images, train, val


def preflight(cfg):
    if cfg.get('mode') not in {'train', 'test'}:
        raise ValueError(
            'mode doit être train ou test dans configs/config.yaml.')
    mode = cfg['mode']
    name = cfg.get('run_name')
    if name is not None and (not isinstance(name, str) or not name
                             or Path(name).name != name or name in {'.', '..'}
                             or '/' in name or '\\' in name):
        raise ValueError(
            'run_name doit être un simple nom de dossier, ou null.')
    if mode == 'train':
        tr = cfg['training']
        for key in ['max_iters', 'val_interval', 'batch_size', 'accumulation']:
            positive(tr[key], f'training.{key}')
        if tr['val_interval'] > tr['max_iters']:
            raise ValueError(
                'val_interval doit être inférieur ou égal à max_iters pour sélectionner le meilleur modèle.'
            )
        if not isinstance(tr['num_workers'], int) or tr['num_workers'] < 0:
            raise ValueError('num_workers doit être un entier positif ou nul.')
        if tr['learning_rate'] <= 0:
            raise ValueError('learning_rate doit être positif.')
        require_file(get_path(cfg, 'training.base_config'),
                     'Configuration Mask2Former de base')
        require_file(get_path(cfg, 'training.start_checkpoint'),
                     'Poids de départ')
        if tr.get('resume_from'):
            require_file(get_path(cfg, 'training.resume_from'),
                         'Checkpoint de reprise')
        images, train, val = training_inputs(cfg)
        if tr['split_strategy'] == 'random':
            print(
                'Conseil : pour des images vidéo proches, utiliser split_strategy: group (docs/DONNEES.md).'
            )
        return dict(mode=mode,
                    images=len(images),
                    train=len(train),
                    validation=len(val))
    test = cfg['test']
    for key in ['materials', 'indicators', 'analysis', 'previews']:
        if type(test[key]) is not bool:
            raise ValueError(
                f'test.{key} doit être true ou false (sans guillemets).')
    if test['indicators'] and not test['materials']:
        raise ValueError(
            'indicators: true exige materials: true pour la perméabilité.')
    if test['analysis'] and not test['indicators']:
        raise ValueError('analysis: true exige indicators: true.')
    images = inspect_images(get_path(cfg, 'test.images_dir'),
                            panorama=test['indicators'])
    require_file(get_path(cfg, 'test.model_config'),
                 'Configuration du modèle sémantique')
    require_file(get_path(cfg, 'test.checkpoint'),
                 'Poids du modèle sémantique')
    if test.get('labels_dir'):
        inspect_labels(images, get_path(cfg, 'test.labels_dir'))
    if test['materials']:
        folder = get_path(cfg, 'bfms.model_dir')
        for name in ['config.json', 'preprocessor_config.json']:
            require_file(folder / name, f'Fichier BFMS {name}')
        if not any((folder / n).is_file() for n in [
                'model.safetensors', 'pytorch_model.bin',
                'model.safetensors.index.json', 'pytorch_model.bin.index.json'
        ]):
            raise ValueError('Poids BFMS absents. Voir docs/MODELES.md.')
        meta = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        from coolpath.taxonomy import BFMS_CLASSES
        labels = meta.get('id2label', {})
        if len(labels) != 43 or any(
                labels.get(str(i)) != label
                for i, label in enumerate(BFMS_CLASSES)):
            raise ValueError(
                'config.json BFMS doit déclarer les 43 noms et indices exacts de taxonomy.py.'
            )
        if cfg['bfms'].get('no_object_policy') != 'notebook':
            raise ValueError(
                'Seule la convention BFMS notebook est implémentée.')
        if not cfg['bfms']['scales']:
            raise ValueError('bfms.scales ne doit pas être vide.')
        for scale in cfg['bfms']['scales']:
            positive(scale, 'bfms.scales')
        require_file(get_path(cfg, 'fusion.ral_table'), 'Table RAL')
        paint = cfg['fusion'].get('paint_albedo')
        if paint is not None and not 0 <= paint <= 1:
            raise ValueError(
                'paint_albedo doit être compris entre 0 et 1, ou null.')
    if test['indicators']:
        positive(cfg['indicators']['n_sectors'], 'indicators.n_sectors')
        if not -90 <= cfg['indicators']['elevation_threshold_deg'] <= 90:
            raise ValueError('Seuil d’élévation hors de [-90,90].')
        if not 0 < cfg['indicators']['azimuth_width_deg'] <= 360:
            raise ValueError('Largeur azimutale hors de ]0,360].')
    if test['analysis']:
        import pandas as pd
        from coolpath.analysis.pipeline import timestamp_images
        from coolpath.thermal.comfypack import load_comfypack
        a = cfg['analysis']
        require_file(get_path(cfg, 'analysis.comfypack_csv'),
                     'Mesures Comfy’Pack')
        if not a['day'] or not a['t0'] or not a['fps']:
            raise ValueError(
                'Configurer analysis.day, t0 et fps pour VOTRE acquisition.')
        if set(a['t0']) != set(a['fps']) or any(v <= 0
                                                for v in a['fps'].values()):
            raise ValueError(
                'Chaque chapitre t0 doit posséder un fps strictement positif.')
        if not a['windows_s'] or any(
                v <= 0 for v in a['windows_s']
        ) or a['reference_window_s'] not in a['windows_s']:
            raise ValueError(
                'Fenêtres positives requises ; reference_window_s doit figurer dans windows_s.'
            )
        stamped = timestamp_images(
            pd.DataFrame({'image': [p.stem for p in images]}), a['t0'],
            a['fps'], a['day'])
        if stamped.ts.isna().any():
            raise ValueError(
                'Horodatage impossible : noms de frames ou chapitres inconnus. Voir docs/DONNEES.md.'
            )
        cp = load_comfypack(get_path(cfg, 'analysis.comfypack_csv'),
                            a['ta_source'])
        needed = {'ts', 'Tg', 'Ta', 'RH', 'WS', 'Kdown'}
        if not needed.issubset(cp.columns):
            raise ValueError(
                f'Colonnes capteurs manquantes : {sorted(needed - set(cp.columns))}'
            )
        if cp.empty or cp.ts.max() < stamped.ts.min() or cp.ts.min(
        ) > stamped.ts.max():
            raise ValueError(
                'Les heures des mesures et des images ne se recouvrent pas.')
    return dict(mode=mode,
                images=len(images),
                materials=test['materials'],
                indicators=test['indicators'],
                analysis=test['analysis'],
                evaluation=bool(test.get('labels_dir')))


def check_environment(cfg):
    import torch
    from mmcv.ops import point_sample  # Verify compiled ops, not only package presence.
    import mmseg
    import mmdet
    if cfg['mode'] == 'train':
        if not torch.cuda.is_available():
            raise ValueError(
                'Le mode train de ce projet requiert un GPU NVIDIA/CUDA. Voir docs/INSTALLATION.md.'
            )
    else:
        device = cfg['test']['device']
        if device != 'cpu' and not device.startswith('cuda'):
            raise ValueError(
                'device doit être cpu ou cuda:0 (ou un autre indice CUDA).')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError(
                    'CUDA indisponible : configurer test.device: cpu ou installer PyTorch CUDA.'
                )
            torch.empty(1, device=device)
        if cfg['test']['materials']:
            import transformers
        if cfg['test']['analysis']:
            from pythermalcomfort.models import utci


def run_train(cfg, run):
    from coolpath.segmentation.training import write_mask2former_config, run_training, select_checkpoint, export_model
    tr = cfg['training']
    _, train, val = training_inputs(cfg)
    labels = get_path(
        cfg,
        'training.labels_dir') if tr['dataset_format'] == 'indexed' else None
    if tr['dataset_format'] == 'cvat':
        from coolpath.data.cvat import convert_cvat_export
        converted = run / 'converted_cvat'
        convert_cvat_export(get_path(cfg, 'training.cvat_export_dir'),
                            get_path(cfg, 'training.images_dir'), converted)
        images = inspect_images(converted / 'images')
        labels = converted / 'labels'
    else:
        images = inspect_images(get_path(cfg, 'training.images_dir'))
    dataset = prepare_indexed_dataset(images, labels, run / 'dataset', train,
                                      val)
    work = run / 'training'
    config = write_mask2former_config(get_path(cfg, 'training.base_config'),
                                      dataset,
                                      work,
                                      get_path(cfg,
                                               'training.start_checkpoint'),
                                      max_iters=tr['max_iters'],
                                      val_interval=tr['val_interval'],
                                      seed=tr['seed'],
                                      batch_size=tr['batch_size'],
                                      num_workers=tr['num_workers'],
                                      accumulation=tr['accumulation'],
                                      learning_rate=tr['learning_rate'],
                                      amp=tr['amp'])
    code = run_training(config,
                        work,
                        resume=False,
                        resume_from=repo_path(tr.get('resume_from'), cfg))
    if code:
        raise RuntimeError(
            f'MMSegmentation a interrompu l’entraînement (code {code}). Consulter les logs dans {work}.'
        )
    checkpoint = select_checkpoint(work)
    exported = export_model(config, checkpoint, run / 'model')
    return dict(train=len(train),
                validation=len(val),
                selected_checkpoint=checkpoint.name,
                model_config=str(exported / 'model_config.py'),
                checkpoint=str(exported / 'model.pth'))


def validate_masks(images, folder, suffix, count):
    import numpy as np
    from PIL import Image
    for p in images:
        mask_path = Path(folder) / f'{p.stem}{suffix}.png'
        require_file(mask_path, 'Masque produit')
        mask = np.asarray(Image.open(mask_path))
        with Image.open(p) as im:
            if mask.shape != (im.height, im.width) or not np.isin(
                    mask, range(count)).all():
                raise ValueError(
                    f'Masque invalide (taille ou indices) : {mask_path}')


def run_test(cfg, run):
    from coolpath.segmentation.semantic import load_mmseg_model, run_inference
    from coolpath.reporting import write_previews, evaluate_masks
    test = cfg['test']
    images_dir = get_path(cfg, 'test.images_dir')
    images = inspect_images(images_dir)
    sem = run / 'semantic' / 'masks'
    model = load_mmseg_model(get_path(cfg, 'test.model_config'),
                             get_path(cfg, 'test.checkpoint'), test['device'])
    from coolpath.taxonomy import CITYSCAPES_CLASSES
    classes = model.dataset_meta.get('classes', [])
    if [n.replace(' ', '_') for n in classes] != CITYSCAPES_CLASSES:
        raise ValueError(
            'Le modèle sémantique doit utiliser les 19 classes Cityscapes dans l’ordre du projet.'
        )
    print('[1] Segmentation sémantique', flush=True)
    run_inference(model, images_dir, sem)
    validate_masks(images, sem, '_sem', 19)
    del model
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = dict(images=len(images))
    if test['previews']:
        write_previews(images, sem, run / 'semantic' / 'previews')
    if test.get('labels_dir'):
        summary['evaluation'] = evaluate_masks(
            images, sem, get_path(cfg, 'test.labels_dir'), run / 'evaluation')
    if test['materials']:
        from coolpath.materials.bfms import run_bfms
        from coolpath.materials.fusion import run_fusion
        print('[2] Matériaux BFMS et fusion', flush=True)
        bfms = run / 'bfms'
        run_bfms(images_dir, bfms, get_path(cfg, 'bfms.model_dir'),
                 test['device'], tuple(cfg['bfms']['scales']))
        validate_masks(images, bfms / 'masks_bfms', '_bfms', 43)
        summary['paint_albedo'] = run_fusion(images_dir, sem,
                                             bfms / 'masks_bfms',
                                             run / 'fusion',
                                             get_path(cfg, 'fusion.ral_table'),
                                             cfg['fusion'].get('paint_albedo'))
    if test['indicators']:
        from coolpath.indicators.morphology import run_indicators
        p = cfg['indicators']
        print('[3] Indicateurs urbains', flush=True)
        run_indicators(images_dir, sem, run / 'fusion' / 'masks_fused',
                       run / 'indicators', p['elevation_threshold_deg'],
                       p['azimuth_width_deg'], p['n_sectors'])
    if test['analysis']:
        from coolpath.analysis.pipeline import run_analysis
        a = cfg['analysis']
        print('[4] Analyse des mesures thermiques', flush=True)
        run_analysis(run / 'indicators' / 'indicateurs_par_image.csv',
                     run / 'fusion' / 'fusion_par_image.csv',
                     get_path(cfg, 'analysis.comfypack_csv'), run / 'analysis',
                     a['t0'], a['fps'], a['day'], a['windows_s'],
                     a['reference_window_s'], a['ta_source'])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='CoolPath : entraîner ou tester votre dataset.')
    parser.add_argument('--config',
                        default='configs/config.yaml',
                        help='Fichier YAML à utiliser')
    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        help='Remplace le mode du YAML pour ce lancement')
    parser.add_argument('--check',
                        action='store_true',
                        help='Vérifier les fichiers sans charger les modèles')
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='Vérifier aussi les bibliothèques et CUDA, sans lancer de calcul')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Afficher la trace complète en cas d’erreur')
    args = parser.parse_args(argv)
    run = None
    try:
        cfg = load_config(args.config)
        if args.mode:
            cfg['mode'] = args.mode
        summary = preflight(cfg)
        print('Entrées vérifiées : ' + json.dumps(summary, ensure_ascii=False),
              flush=True)
        if args.check and not args.check_env:
            print(
                'Ce contrôle ne charge pas les poids et ne valide pas leur compatibilité interne.'
            )
            return 0
        check_environment(cfg)
        if args.check_env:
            print(
                'Environnement vérifié ; aucun entraînement ni inférence lancé.'
            )
            return 0
        name = cfg.get('run_name') or datetime.now(
            timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        run = get_path(cfg, 'output_dir') / f'{cfg["mode"]}_{name}'
        run.mkdir(parents=True, exist_ok=False)
        (run / 'config_used.yaml').write_text(yaml.safe_dump(
            cfg, allow_unicode=True, sort_keys=False),
                                              encoding='utf-8')
        versions = {'python': sys.version}
        for package in [
                'numpy', 'torch', 'torchvision', 'mmcv', 'mmengine',
                'mmsegmentation', 'mmdet', 'transformers'
        ]:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                pass
        (run / 'environment.json').write_text(json.dumps(versions, indent=2),
                                              encoding='utf-8')
        (run / 'status.json').write_text(json.dumps({'status': 'running'}),
                                         encoding='utf-8')
        result = run_train(cfg, run) if cfg['mode'] == 'train' else run_test(
            cfg, run)
        (run / 'summary.json').write_text(json.dumps(result,
                                                     indent=2,
                                                     ensure_ascii=False),
                                          encoding='utf-8')
        (run / 'status.json').write_text(json.dumps({'status': 'completed'}),
                                         encoding='utf-8')
        print(f'Terminé. Résultats : {run}')
        if cfg['mode'] == 'train':
            print('Pour tester, choisir mode: test et renseigner :\n' +
                  yaml.safe_dump(
                      {
                          'test': {
                              k: result[k]
                              for k in ['model_config', 'checkpoint']
                          }
                      },
                      sort_keys=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if run is not None and run.is_dir():
            # Only a newly created run may receive failure metadata.
            if (run / 'status.json').is_file() and not isinstance(
                    exc, FileExistsError):
                (run / 'status.json').write_text(json.dumps(
                    {
                        'status': 'failed',
                        'error': str(exc)
                    },
                    ensure_ascii=False),
                                                 encoding='utf-8')
                (run / 'error.log').write_text(traceback.format_exc(),
                                               encoding='utf-8')
        print(f'Erreur : {exc}', file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 2
