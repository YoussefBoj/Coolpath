"""Tests of public input contracts and orchestration; no model weights required."""
import ast
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from coolpath.cli import main, preflight, run_test
from coolpath.config import get_path, load_config
from coolpath.data.dataset import inspect_images, inspect_labels, make_splits, prepare_indexed_dataset
from coolpath.reporting import evaluate_masks
from coolpath.segmentation.training import write_mask2former_config, select_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def example(tmp_path):
    cfg = load_config(ROOT / 'configs/config.yaml')
    cfg['_root'] = str(tmp_path)
    cfg['test'].update(images_dir='images',
                       model_config='model.py',
                       checkpoint='model.pth',
                       device='cpu')
    (tmp_path / 'model.py').write_text('# placeholder for preflight only')
    (tmp_path / 'model.pth').touch()
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    for i in range(3):
        Image.new('RGB', (32, 16),
                  (100, 120, 130)).save(tmp_path / 'images' / f'image{i}.png')
        mask = np.zeros((16, 32), dtype=np.uint8)
        mask[:8] = 10
        Image.fromarray(mask).save(tmp_path / 'labels' / f'image{i}.png')
    return cfg


def test_paths_independent_of_cwd(tmp_path, monkeypatch):
    cfg = load_config(ROOT / 'configs/config.yaml')
    monkeypatch.chdir(tmp_path)
    assert get_path(cfg, 'test.images_dir') == ROOT / 'data/test/images'


def test_preflight_minimal_test(tmp_path):
    cfg = example(tmp_path)
    assert preflight(cfg)['images'] == 3


def test_missing_weights_actionable(tmp_path):
    cfg = example(tmp_path)
    (tmp_path / 'model.pth').unlink()
    with pytest.raises(ValueError, match='Poids'):
        preflight(cfg)


def test_indices_and_shape(tmp_path):
    cfg = example(tmp_path)
    images = inspect_images(tmp_path / 'images')
    Image.new('RGB', (32, 16)).save(tmp_path / 'labels/image0.png')
    with pytest.raises(ValueError, match='un canal'):
        inspect_labels(images, tmp_path / 'labels')
    Image.fromarray(np.full(
        (16, 32), 24, dtype=np.uint8)).save(tmp_path / 'labels/image0.png')
    with pytest.raises(ValueError, match='indices invalides'):
        inspect_labels(images, tmp_path / 'labels')


def test_group_split_no_leakage():
    groups = {f'image{i}': f'group{i//3}' for i in range(12)}
    train, val = make_splits(list(groups), strategy='group', groups=groups)
    assert not {groups[s] for s in train} & {groups[s] for s in val}
    assert (train, val) == make_splits(list(groups),
                                       strategy='group',
                                       groups=groups)


def test_random_unique_names_have_nonempty_train():
    train, val = make_splits(['a', 'b', 'c', 'd'])
    assert train and val and not set(train) & set(val)


def test_explicit_split_rejects_overlap(tmp_path):
    (tmp_path / 'train.txt').write_text('a\nb\n')
    (tmp_path / 'val.txt').write_text('b\n')
    with pytest.raises(ValueError, match='disjoints'):
        make_splits(['a', 'b'], strategy='files', split_dir=tmp_path)


def test_empty_and_duplicate_inputs(tmp_path):
    with pytest.raises(ValueError, match='Aucune image'):
        inspect_images(tmp_path)
    Image.new('RGB', (8, 4)).save(tmp_path / 'a.jpg')
    Image.new('RGB', (8, 4)).save(tmp_path / 'a.png')
    with pytest.raises(ValueError, match='unique'):
        inspect_images(tmp_path)


def test_pipeline_dependencies_and_geometry(tmp_path):
    cfg = example(tmp_path)
    cfg['test']['indicators'] = True
    with pytest.raises(ValueError, match='materials: true'):
        preflight(cfg)
    Image.new('RGB', (10, 10)).save(tmp_path / 'images/image0.png')
    with pytest.raises(ValueError, match='2:1'):
        inspect_images(tmp_path / 'images', panorama=True)


def test_normalize_mixed_extensions_and_preserve_masks(tmp_path):
    cfg = example(tmp_path)
    Image.new('RGB', (32, 16)).save(tmp_path / 'images/image3.jpg')
    Image.new('L', (32, 16), 9).save(tmp_path / 'labels/image3.png')
    images = inspect_images(tmp_path / 'images')
    train, val = make_splits([p.stem for p in images])
    out = prepare_indexed_dataset(images, tmp_path / 'labels',
                                  tmp_path / 'prepared', train, val)
    assert len(list((out / 'images').glob('*.png'))) == 4
    assert np.asarray(Image.open(out / 'labels/image3.png')).max() == 9


def test_generated_training_config_replaces_cityscapes_dataset(tmp_path):
    path = write_mask2former_config(tmp_path / 'base.py',
                                    tmp_path / 'dataset',
                                    tmp_path / 'work',
                                    tmp_path / 'start.pth',
                                    seed=7,
                                    max_iters=1,
                                    val_interval=1)
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(path.read_text()).body
    }
    loader = values['train_dataloader']
    assert loader['_delete_'] is True
    assert loader['dataset']['reduce_zero_label'] is False
    assert loader['dataset']['type'] == 'BaseSegDataset'
    assert values['randomness']['seed'] == 7
    assert values['optim_wrapper']['type'] == 'AmpOptimWrapper'
    assert all(s['end'] > s['begin'] for s in values['param_scheduler'])


def test_checkpoint_numeric_order(tmp_path):
    for i in [750, 1000, 250]:
        (tmp_path / f'best_mIoU_iter_{i}.pth').touch()
    assert select_checkpoint(tmp_path).name == 'best_mIoU_iter_1000.pth'


def test_false_positive_class_penalizes_standard_miou(tmp_path):
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'masks').mkdir()
    p = tmp_path / 'image.png'
    Image.new('RGB', (2, 1)).save(p)
    Image.fromarray(np.array([[0, 0]], dtype=np.uint8)).save(
        tmp_path / 'labels/image.png')
    Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(
        tmp_path / 'masks/image_sem.png')
    metrics = evaluate_masks([p], tmp_path / 'masks', tmp_path / 'labels',
                             tmp_path / 'scores')
    assert metrics['mIoU_pct'] == 25
    assert metrics['mIoU_GT_present_pct'] == 50


def test_cvat_preflight_and_conversion(tmp_path):
    cfg = example(tmp_path)
    cvat = tmp_path / 'cvat'
    (cvat / 'SegmentationClass').mkdir(parents=True)
    (cvat /
     'labelmap.txt').write_text('background:0,0,0::\nsky:70,130,180::\n')
    for i in range(3):
        Image.new('RGB', (32, 16), (70, 130, 180)).save(
            cvat / 'SegmentationClass' / f'image{i}.png')
    cfg['mode'] = 'train'
    cfg['training'].update(dataset_format='cvat',
                           cvat_export_dir='cvat',
                           images_dir='images',
                           base_config='model.py',
                           start_checkpoint='model.pth')
    assert preflight(cfg)['train'] == 2


def test_inference_routing_and_reports(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from coolpath.segmentation import semantic
    from coolpath.taxonomy import CITYSCAPES_CLASSES
    cfg = example(tmp_path)
    cfg['test']['labels_dir'] = 'labels'
    monkeypatch.setitem(
        sys.modules, 'torch',
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(
        semantic, 'load_mmseg_model', lambda *a: SimpleNamespace(
            dataset_meta={'classes': CITYSCAPES_CLASSES}))
    # Only the model prediction boundary is replaced. Real IO/metrics/previews run.
    monkeypatch.setattr(
        semantic, 'predict_semantic', lambda model, p: np.asarray(
            Image.open(tmp_path / 'labels' / f'{Path(p).stem}.png')))
    result = run_test(cfg, tmp_path / 'result')
    assert result['evaluation']['mIoU_pct'] == 100
    assert len(list(
        (tmp_path / 'result/semantic/previews').glob('*.jpg'))) == 3
    assert not (tmp_path / 'result/bfms').exists()


def test_full_morphology_pipeline_with_synthetic_model_outputs(
        tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from coolpath.segmentation import semantic
    from coolpath.materials import bfms
    from coolpath.taxonomy import CITYSCAPES_CLASSES
    cfg = example(tmp_path)
    cfg['test'].update(materials=True, indicators=True)
    cfg['fusion'].update(ral_table=str(ROOT / 'resources/ral_albedo.csv'),
                         paint_albedo=.45)
    monkeypatch.setitem(
        sys.modules, 'torch',
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(
        semantic, 'load_mmseg_model', lambda *a: SimpleNamespace(
            dataset_meta={'classes': CITYSCAPES_CLASSES}))
    monkeypatch.setattr(
        semantic, 'predict_semantic', lambda model, p: np.asarray(
            Image.open(tmp_path / 'labels' / f'{Path(p).stem}.png')))
    monkeypatch.setattr(bfms, 'load_bfms', lambda *a: (None, None, 'cpu'))
    monkeypatch.setattr(
        bfms, 'predict_bfms',
        lambda m, p, d, im, s: np.full(im.shape[:2], 9, dtype=np.uint8))
    run_test(cfg, tmp_path / 'result')
    import pandas as pd
    table = pd.read_csv(tmp_path /
                        'result/indicators/indicateurs_par_image.csv')
    assert len(table) == 3
    assert np.allclose(table['SVF'], 1, atol=.02)
    assert np.allclose(table['taux_impermeable_pct'], 100)


def test_cli_check_does_not_load_gpu_or_create_output(tmp_path):
    import yaml
    cfg = example(tmp_path)
    cfg['project_root'] = str(tmp_path)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert main(['--config', str(path), '--check']) == 0
    assert not (tmp_path / 'outputs').exists()


def test_cli_existing_run_is_not_modified(tmp_path, monkeypatch):
    import yaml
    import coolpath.cli as cli
    cfg = example(tmp_path)
    cfg.update(project_root=str(tmp_path), run_name='existing')
    folder = tmp_path / 'outputs/test_existing'
    folder.mkdir(parents=True)
    (folder / 'status.json').write_text('original')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(cli, 'check_environment', lambda cfg: None)
    assert main(['--config', str(path)]) == 2
    assert (folder / 'status.json').read_text() == 'original'


def test_train_mode_prepares_data_and_exports_selected_model(
        tmp_path, monkeypatch):
    from coolpath.cli import run_train
    from coolpath.segmentation import training
    cfg = example(tmp_path)
    cfg['mode'] = 'train'
    cfg['training'].update(images_dir='images',
                           labels_dir='labels',
                           base_config='model.py',
                           start_checkpoint='model.pth',
                           max_iters=10,
                           val_interval=5)

    def trainer(config, work, **kwargs):
        assert (tmp_path / 'train/dataset/splits/train.txt').exists()
        (work / 'best_mIoU_iter_5.pth').write_bytes(b'test checkpoint')
        return 0

    def exporter(config, checkpoint, destination):
        assert checkpoint.name == 'best_mIoU_iter_5.pth'
        destination.mkdir()
        (destination /
         'model_config.py').write_text('# fake model config for routing test')
        (destination / 'model.pth').write_bytes(checkpoint.read_bytes())
        return destination

    monkeypatch.setattr(training, 'run_training', trainer)
    monkeypatch.setattr(training, 'export_model', exporter)
    result = run_train(cfg, tmp_path / 'train')
    assert result['train'] == 2 and result['validation'] == 1
    assert Path(result['checkpoint']).exists()


def test_explicit_resume_uses_mmseg_supported_arguments(tmp_path, monkeypatch):
    from coolpath.segmentation import training
    captured = []
    monkeypatch.setattr(training, 'find_mmseg_tool',
                        lambda name: tmp_path / name)
    monkeypatch.setattr(training.subprocess, 'call',
                        lambda cmd: captured.extend(cmd) or 0)
    training.run_training(tmp_path / 'model.py',
                          tmp_path / 'work',
                          resume_from=tmp_path / 'iter_10.pth')
    assert '--resume' in captured
    # MMSegmentation 1.2.2 --resume is a boolean switch; load_from is a cfg override.
    assert '--cfg-options' in captured
    assert any(
        v.startswith('load_from=') and 'iter_10.pth' in v for v in captured)
