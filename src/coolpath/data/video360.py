from __future__ import annotations
from pathlib import Path
import shutil
import cv2
import numpy as np
from coolpath.utils.io import ensure_dir

def laplacian_variance(bgr: np.ndarray, resize=(768, 384)) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def extract_candidates(video_path: str | Path, output_dir: str | Path,
                       candidate_fps: float = 0.5, blur_threshold: float = 100.0,
                       jpeg_quality: int = 95) -> tuple[list[Path], int]:
    video_path, output_dir = Path(video_path), ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Vidéo illisible: {video_path}')
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(src_fps / candidate_fps))
    kept, rejected, idx = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            if laplacian_variance(frame) >= blur_threshold:
                out = output_dir / f'{video_path.stem}__f{idx:06d}.jpg'
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                kept.append(out)
            else:
                rejected += 1
        idx += 1
    cap.release()
    return kept, rejected

def color_histogram(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img = cv2.resize(img, (256, 256))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8,8,8], [0,256]*3)
    return cv2.normalize(hist, hist).flatten()

def greedy_diverse(frames: list[Path], k: int) -> list[Path]:
    if len(frames) <= k: return list(frames)
    hists = {f: color_histogram(f) for f in frames}
    selected = [frames[0]]
    while len(selected) < k:
        best_f, best_d = None, -1.0
        for f in frames:
            if f in selected: continue
            d = min(cv2.compareHist(hists[f], hists[s], cv2.HISTCMP_BHATTACHARYYA) for s in selected)
            if d > best_d: best_d, best_f = d, f
        selected.append(best_f)
    return selected

def deduplicate(frames: list[Path], threshold: float = 0.15) -> list[Path]:
    hists = {f: color_histogram(f) for f in frames}
    final = []
    for f in frames:
        if not any(cv2.compareHist(hists[f], hists[g], cv2.HISTCMP_BHATTACHARYYA) < threshold for g in final):
            final.append(f)
    return final

def copy_selected(frames: list[Path], output_dir: str | Path, clear=True) -> list[Path]:
    out = ensure_dir(output_dir)
    if clear:
        for p in out.glob('*.jpg'): p.unlink()
    copied=[]
    for p in sorted(frames):
        dst=out/p.name; shutil.copy2(p,dst); copied.append(dst)
    return copied
