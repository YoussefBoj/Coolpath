#!/usr/bin/env python
import argparse
from coolpath.config import load_config, get_path
from coolpath.segmentation.semantic import load_mmseg_model
from coolpath.data.preannotation import export_cvat_polygons, write_cvat_labels
from coolpath.utils.io import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pipeline.yaml')
    ap.add_argument('--model-config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = load_mmseg_model(args.model_config, args.checkpoint, args.device)
    out = get_path(cfg, 'paths.frames_to_annotate_dir').parent
    export_cvat_polygons(
        model, list_images(get_path(cfg, 'paths.frames_to_annotate_dir')),
        out / 'cvat_preannotations.xml', out / 'preview_results')
    write_cvat_labels(out / 'cvat_labels_cityscapes19.json')


if __name__ == '__main__': main()
