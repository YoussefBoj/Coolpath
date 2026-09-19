"""MMSegmentation training and portable model export."""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
import sys
from coolpath.taxonomy import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE


def find_mmseg_tool(name: str) -> Path:
    import mmseg
    root = Path(mmseg.__file__).parent
    direct = root / '.mim' / 'tools' / name
    if direct.exists():
        return direct
    candidates = list(root.rglob(f'tools/{name}'))
    if not candidates:
        raise FileNotFoundError(
            f'Outil MMSeg introuvable : {name}. Voir docs/INSTALLATION.md.')
    return candidates[0]


def write_mask2former_config(base_config,
                             dataset_dir,
                             work_dir,
                             start_checkpoint,
                             image_suffix='.png',
                             max_iters=5000,
                             val_interval=250,
                             seed=42,
                             batch_size=2,
                             num_workers=0,
                             accumulation=4,
                             learning_rate=5e-5,
                             amp=True):
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='RandomResize',
             scale=(2048, 1024),
             ratio_range=(.5, 1.5),
             keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=.75),
        dict(type='RandomFlip', prob=.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs')
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='PackSegInputs')
    ]
    meta = dict(classes=tuple(CITYSCAPES_CLASSES),
                palette=CITYSCAPES_PALETTE.tolist())

    def loader(split, batch, pipeline):
        return dict(_delete_=True,
                    batch_size=batch,
                    num_workers=num_workers,
                    persistent_workers=num_workers > 0,
                    sampler=dict(type='InfiniteSampler'
                                 if split == 'train' else 'DefaultSampler',
                                 shuffle=split == 'train'),
                    dataset=dict(type='BaseSegDataset',
                                 data_root=str(Path(dataset_dir).resolve()),
                                 data_prefix=dict(img_path='images',
                                                  seg_map_path='labels'),
                                 img_suffix=image_suffix,
                                 seg_map_suffix='.png',
                                 reduce_zero_label=False,
                                 ann_file=f'splits/{split}.txt',
                                 metainfo=meta,
                                 pipeline=pipeline))

    warmup = min(200, max_iters - 1)
    schedulers = []
    if warmup:
        schedulers.append(
            dict(type='LinearLR',
                 start_factor=.1,
                 by_epoch=False,
                 begin=0,
                 end=warmup))
    schedulers.append(
        dict(type='PolyLR',
             eta_min=1e-6,
             power=.9,
             begin=warmup,
             end=max_iters,
             by_epoch=False))
    cfg = dict(
        _base_=str(Path(base_config).resolve()),
        train_dataloader=loader('train', batch_size, train_pipeline),
        val_dataloader=loader('val', 1, test_pipeline),
        test_dataloader=loader('val', 1, test_pipeline),
        train_pipeline=train_pipeline,
        test_pipeline=test_pipeline,
        # Do not download a backbone when the complete starting checkpoint is supplied.
        model=dict(backbone=dict(frozen_stages=2, init_cfg=None),
                   decode_head=dict(num_classes=19)),
        load_from=str(Path(start_checkpoint).resolve()),
        optim_wrapper=dict(type='AmpOptimWrapper' if amp else 'OptimWrapper',
                           optimizer=dict(type='AdamW',
                                          lr=learning_rate,
                                          weight_decay=.05),
                           accumulative_counts=accumulation),
        param_scheduler=schedulers,
        train_cfg=dict(_delete_=True,
                       type='IterBasedTrainLoop',
                       max_iters=max_iters,
                       val_interval=val_interval),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),
        val_evaluator=dict(_delete_=True,
                           type='IoUMetric',
                           iou_metrics=['mIoU']),
        test_evaluator=dict(_delete_=True,
                            type='IoUMetric',
                            iou_metrics=['mIoU']),
        default_hooks=dict(checkpoint=dict(type='CheckpointHook',
                                           by_epoch=False,
                                           interval=val_interval,
                                           max_keep_ckpts=3,
                                           save_best='mIoU',
                                           save_last=True),
                           logger=dict(type='LoggerHook', interval=25)),
        randomness=dict(seed=seed),
        work_dir=str(work_dir))
    if amp:
        cfg['optim_wrapper']['loss_scale'] = 'dynamic'
    path = work_dir / 'mask2former_coolpath_final.py'
    path.write_text('\n\n'.join(f'{k} = {v!r}' for k, v in cfg.items()) + '\n',
                    encoding='utf-8')
    return path


def run_training(config_path, work_dir, resume=True, resume_from=None):
    cmd = [
        sys.executable,
        str(find_mmseg_tool('train.py')),
        str(config_path), '--work-dir',
        str(work_dir)
    ]
    if resume_from:
        cmd.extend([
            '--resume', '--cfg-options',
            f'load_from={Path(resume_from).resolve().as_posix()}'
        ])
    elif resume and (Path(work_dir) / 'last_checkpoint').exists():
        cmd.append('--resume')
    return subprocess.call(cmd)


def select_checkpoint(work_dir):
    work_dir = Path(work_dir)

    def iteration(p):
        match = re.search(r'iter_(\d+)', p.name)
        return int(match.group(1)) if match else -1

    best = sorted(work_dir.glob('best_mIoU_iter_*.pth'), key=iteration)
    if best:
        return best[-1]
    last = work_dir / 'last_checkpoint'
    if last.exists():
        p = Path(last.read_text(encoding='utf-8').strip())
        if not p.is_absolute():
            p = work_dir / p
        if p.is_file():
            return p
    candidates = sorted(work_dir.glob('iter_*.pth'), key=iteration)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError('Aucun checkpoint produit par l’entraînement.')


def export_model(config_path, checkpoint, destination):
    from mmengine.config import Config
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config = Config.fromfile(str(config_path))
    # Export only inference requirements; resolve _base_ and remove training data paths.
    inference = Config(
        dict(model=config.model,
             test_pipeline=config.test_pipeline,
             default_scope=config.get('default_scope', 'mmseg'),
             custom_imports=config.get(
                 'custom_imports',
                 dict(imports=['mmdet.models'], allow_failed_imports=False))))
    inference.dump(str(destination / 'model_config.py'))
    shutil.copy2(checkpoint, destination / 'model.pth')
    return destination
