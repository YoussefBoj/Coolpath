#!/usr/bin/env python
import argparse
from pathlib import Path
from coolpath.config import load_config,get_path
from coolpath.segmentation.semantic import load_mmseg_model,run_inference

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/pipeline.yaml'); ap.add_argument('--checkpoint'); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args(); cfg=load_config(args.config); wd=get_path(cfg,'paths.semantic_work_dir'); checkpoint=Path(args.checkpoint) if args.checkpoint else sorted(wd.glob('best_mIoU_iter_*.pth'))[-1]; model=load_mmseg_model(wd/'mask2former_coolpath_final.py',checkpoint,args.device); run_inference(model,get_path(cfg,'paths.all_images_dir'),get_path(cfg,'paths.semantic_masks_dir'))
if __name__=='__main__': main()
