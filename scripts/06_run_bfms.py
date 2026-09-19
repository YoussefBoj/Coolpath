#!/usr/bin/env python
import argparse
from coolpath.config import load_config, get_path
from coolpath.materials.bfms import run_bfms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pipeline.yaml')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_bfms(get_path(cfg, 'paths.all_images_dir'),
             get_path(cfg, 'paths.bfms_output_dir'),
             get_path(cfg, 'paths.bfms_model_dir'), args.device,
             tuple(cfg['bfms']['scales']))


if __name__ == '__main__': main()
