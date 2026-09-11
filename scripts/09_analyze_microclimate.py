#!/usr/bin/env python
import argparse
from coolpath.config import load_config,get_path
from coolpath.analysis.pipeline import run_analysis

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/pipeline.yaml'); args=ap.parse_args(); cfg=load_config(args.config); a=cfg['analysis']; result=run_analysis(get_path(cfg,'paths.indicators_output_dir')/'indicateurs_par_image.csv',get_path(cfg,'paths.fusion_output_dir')/'fusion_par_image.csv',get_path(cfg,'paths.comfypack_csv'),get_path(cfg,'paths.analysis_output_dir'),a['t0'],a['fps'],a['day'],a['windows_s'],a['reference_window_s'],a['ta_source']); print(result)
if __name__=='__main__': main()
