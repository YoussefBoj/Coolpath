#!/usr/bin/env python
import argparse
from coolpath.config import load_config,get_path
from coolpath.indicators.morphology import run_indicators

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/pipeline.yaml'); args=ap.parse_args(); cfg=load_config(args.config); p=cfg['indicators']; run_indicators(get_path(cfg,'paths.all_images_dir'),get_path(cfg,'paths.semantic_masks_dir'),get_path(cfg,'paths.fusion_output_dir')/'masks_fused',get_path(cfg,'paths.indicators_output_dir'),p['elevation_threshold_deg'],p['azimuth_width_deg'])
if __name__=='__main__': main()
