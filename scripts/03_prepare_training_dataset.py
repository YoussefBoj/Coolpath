#!/usr/bin/env python
import argparse
from coolpath.config import load_config, get_path
from coolpath.data.cvat import convert_cvat_export, grouped_train_val_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pipeline.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = get_path(cfg, 'paths.dataset_dir')
    info = convert_cvat_export(get_path(cfg, 'paths.cvat_export_dir'),
                               get_path(cfg, 'paths.frames_to_annotate_dir'),
                               ds)
    train, val = grouped_train_val_split(ds / 'images', ds,
                                         cfg['training']['val_fraction'],
                                         cfg['training']['seed'])
    print(info)
    print('train', len(train), 'val', len(val))


if __name__ == '__main__': main()
