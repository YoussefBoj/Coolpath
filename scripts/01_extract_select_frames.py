#!/usr/bin/env python
import argparse
from pathlib import Path
from coolpath.config import load_config, get_path
from coolpath.data.video360 import extract_candidates, greedy_diverse, deduplicate, copy_selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pipeline.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    video_dir = get_path(cfg, 'paths.raw_videos_dir')
    all_dir = get_path(cfg, 'paths.all_images_dir')
    ann_dir = get_path(cfg, 'paths.frames_to_annotate_dir')
    s = cfg['selection']
    selected = []
    for filename, quota in s['quotas'].items():
        frames, rejected = extract_candidates(video_dir / filename, all_dir,
                                              s['candidate_fps'],
                                              s['blur_threshold'])
        chosen = greedy_diverse(frames, int(quota))
        selected.extend(chosen)
        print(filename, len(frames), 'candidates,', rejected, 'floues,',
              len(chosen), 'retenues')
    final = deduplicate(selected, s['dedup_threshold'])
    copy_selected(final, ann_dir)
    print(f'{len(final)} images prêtes pour CVAT: {ann_dir}')


if __name__ == '__main__': main()
