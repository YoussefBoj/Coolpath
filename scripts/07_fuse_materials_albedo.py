#!/usr/bin/env python
import argparse
from coolpath.config import load_config, get_path
from coolpath.materials.fusion import run_fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pipeline.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    paint = run_fusion(get_path(cfg, 'paths.all_images_dir'),
                       get_path(cfg, 'paths.semantic_masks_dir'),
                       get_path(cfg, 'paths.bfms_output_dir') / 'masks_bfms',
                       get_path(cfg, 'paths.fusion_output_dir'),
                       get_path(cfg, 'paths.ral_table'))
    print('Albédo peinture fixe:', paint)


if __name__ == '__main__': main()
