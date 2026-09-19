#!/usr/bin/env python
import argparse
from coolpath.config import load_config, get_path, repo_path
from coolpath.segmentation.training import write_mask2former_config, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pipeline.yaml')
    ap.add_argument('--no-resume', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    tr = cfg['training']
    ds = get_path(cfg, 'paths.dataset_dir')
    wd = get_path(cfg, 'paths.semantic_work_dir')
    suffix = next(p.suffix.lower() for p in (ds / 'images').iterdir()
                  if p.is_file())
    c = write_mask2former_config(repo_path(tr['base_config'], cfg), ds, wd,
                                 repo_path(tr['start_checkpoint'], cfg),
                                 suffix, tr['max_iters'], tr['val_interval'])
    raise SystemExit(run_training(c, wd, not args.no_resume))


if __name__ == '__main__': main()
