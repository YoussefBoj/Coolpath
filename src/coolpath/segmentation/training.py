from __future__ import annotations
from pathlib import Path
import subprocess, sys
from coolpath.taxonomy import CITYSCAPES_CLASSES
from coolpath.utils.io import ensure_dir

def find_mmseg_tool(name: str) -> Path:
    import mmseg
    root=Path(mmseg.__file__).parent
    direct=root/'.mim'/'tools'/name
    if direct.exists(): return direct
    candidates=list(root.rglob(f'tools/{name}'))
    if not candidates: raise FileNotFoundError(f'Outil MMSeg introuvable: {name}')
    return candidates[0]

def write_mask2former_config(base_config, dataset_dir, work_dir, start_checkpoint,
                             image_suffix='.jpg', max_iters=5000, val_interval=250) -> Path:
    work_dir=ensure_dir(work_dir); path=work_dir/'mask2former_coolpath_final.py'
    classes=','.join(repr(x) for x in CITYSCAPES_CLASSES)
    content=f'''_base_ = r"{Path(base_config).as_posix()}"\n\ndataset_type = "BaseSegDataset"\ndata_root = r"{Path(dataset_dir).as_posix()}"\nmetainfo = dict(classes=({classes},))\n\ntrain_pipeline = [\n dict(type="LoadImageFromFile"), dict(type="LoadAnnotations"),\n dict(type="RandomResize", scale=(2048,1024), ratio_range=(0.5,1.5), keep_ratio=True),\n dict(type="RandomCrop", crop_size=(512,1024), cat_max_ratio=0.75),\n dict(type="RandomFlip", prob=0.5), dict(type="PhotoMetricDistortion"), dict(type="PackSegInputs"),\n]\ntest_pipeline = [dict(type="LoadImageFromFile"), dict(type="Resize", scale=(2048,1024), keep_ratio=True), dict(type="LoadAnnotations"), dict(type="PackSegInputs")]\ntrain_dataloader = dict(batch_size=2,num_workers=4,persistent_workers=True,dataset=dict(type=dataset_type,data_root=data_root,data_prefix=dict(img_path="images",seg_map_path="labels"),img_suffix="{image_suffix}",seg_map_suffix=".png",ann_file="splits/train.txt",metainfo=metainfo,pipeline=train_pipeline))\nval_dataloader = dict(batch_size=1,num_workers=2,dataset=dict(type=dataset_type,data_root=data_root,data_prefix=dict(img_path="images",seg_map_path="labels"),img_suffix="{image_suffix}",seg_map_suffix=".png",ann_file="splits/val.txt",metainfo=metainfo,pipeline=test_pipeline))\ntest_dataloader = val_dataloader\nload_from = r"{Path(start_checkpoint).as_posix()}"\nmodel = dict(backbone=dict(frozen_stages=2))\noptim_wrapper = dict(optimizer=dict(type="AdamW",lr=5e-5,weight_decay=0.05),accumulative_counts=4)\nparam_scheduler=[dict(type="LinearLR",start_factor=0.1,by_epoch=False,begin=0,end=200),dict(type="PolyLR",eta_min=1e-6,power=0.9,begin=200,end={max_iters},by_epoch=False)]\ntrain_cfg=dict(type="IterBasedTrainLoop",max_iters={max_iters},val_interval={val_interval})\nval_cfg=dict(type="ValLoop")\ntest_cfg=dict(type="TestLoop")\ndefault_hooks=dict(checkpoint=dict(type="CheckpointHook",by_epoch=False,interval={val_interval},max_keep_ckpts=3,save_best="mIoU",save_last=True),logger=dict(type="LoggerHook",interval=25))\nrandomness=dict(seed=42)\ntta_model=dict(type="SegTTAModel")\nimg_ratios=[0.75,1.0,1.25]\ntta_pipeline=[dict(type="LoadImageFromFile",backend_args=None),dict(type="TestTimeAug",transforms=[[dict(type="Resize",scale_factor=r,keep_ratio=True) for r in img_ratios],[dict(type="RandomFlip",prob=0.,direction="horizontal"),dict(type="RandomFlip",prob=1.,direction="horizontal")],[dict(type="LoadAnnotations")],[dict(type="PackSegInputs")]])]\nwork_dir=r"{work_dir.as_posix()}"\n'''
    path.write_text(content,encoding='utf-8'); return path

def run_training(config_path, work_dir, resume=True) -> int:
    tool=find_mmseg_tool('train.py'); cmd=[sys.executable,str(tool),str(config_path),'--work-dir',str(work_dir)]
    if resume and (Path(work_dir)/'last_checkpoint').exists(): cmd.append('--resume')
    return subprocess.call(cmd)
