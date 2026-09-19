from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, pandas as pd
from coolpath.taxonomy import BFMS_NAME_TO_ID as B, IGNORE_INDEX
from coolpath.materials.albedo import build_albedo_lut, load_ral_table, nearest_ral, shadow_correct_rgb, EXCLUDED_SEMANTIC
from coolpath.utils.io import ensure_dir, list_images

GATING = {
    0: ({'Asphalt', 'Cement/Concrete', 'Natural Stone'}, 'Asphalt'),
    1: ({'Cement/Concrete', 'Asphalt', 'Natural Stone',
         'Brick'}, 'Cement/Concrete'),
    2: ({
        'Brick', 'Carved brick', 'Cement/Concrete', 'Paint/Coating/Plaster',
        'Glass', 'Mirror', 'Window screen', 'Windows with metal fences',
        'Metal', 'Wood/Bamboo', 'Natural Stone', 'Roof tile', 'Ceramic'
    }, 'Paint/Coating/Plaster'),
    3: ({
        'Brick', 'Cement/Concrete', 'Natural Stone', 'Paint/Coating/Plaster',
        'Metal'
    }, 'Cement/Concrete'),
    4: ({'Metal', 'Wood/Bamboo', 'Windows with metal fences',
         'Brick'}, 'Metal'),
    5: ({'Metal', 'Wood/Bamboo', 'Cement/Concrete'}, 'Metal'),
    6: ({'Metal', 'Glass', 'Plastic, clear'}, 'Metal'),
    7: ({'Metal', 'Paint/Coating/Plaster'}, 'Metal'),
    8: ({'Foliage', 'Tree'}, 'Foliage'),
    9: ({'Soil/Mud', 'Foliage', 'Sand'}, 'Soil/Mud'),
    10: ({'Sky'}, 'Sky'),
    11: ({'Skin/Lips', 'Hair', 'Fabric/Cloth', 'Leather'}, 'Fabric/Cloth'),
    12: ({'Skin/Lips', 'Hair', 'Fabric/Cloth', 'Leather'}, 'Fabric/Cloth'),
    13: ({'Metal', 'Glass'}, 'Metal'),
    14: ({'Metal', 'Glass'}, 'Metal'),
    15: ({'Metal', 'Glass'}, 'Metal'),
    16: ({'Metal', 'Glass'}, 'Metal'),
    17: ({'Metal'}, 'Metal'),
    18: ({'Metal'}, 'Metal')
}


def build_gate_lut() -> np.ndarray:
    lut = np.zeros((19, 43), dtype=np.uint8)
    for sem, (allowed, default) in GATING.items():
        allowed_ids = {B[x] for x in allowed}
        default_id = B[default]
        for mat in range(43):
            lut[sem, mat] = mat if mat in allowed_ids else default_id
    return lut


def estimate_global_paint_albedo(images,
                                 semantic_dir,
                                 bfms_dir,
                                 ral_csv,
                                 percentile=95):
    gate = build_gate_lut()
    rgb_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    paint = B['Paint/Coating/Plaster']
    for p in images:
        sem = cv2.imread(str(Path(semantic_dir) / f'{p.stem}_sem.png'),
                         cv2.IMREAD_GRAYSCALE)
        bfms = cv2.imread(str(Path(bfms_dir) / f'{p.stem}_bfms.png'),
                          cv2.IMREAD_GRAYSCALE)
        if sem is None or bfms is None: continue
        valid = sem != IGNORE_INDEX
        fused = np.zeros_like(bfms)
        fused[valid] = gate[sem[valid], bfms[valid]]
        sel = valid & (fused == paint)
        if not sel.any(): continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)[sel]
        corrected = shadow_correct_rgb(rgb, percentile)
        rgb_sum += corrected.sum(axis=0)
        pixel_count += len(corrected)
    if not pixel_count:
        return .45, {'reason': 'no_paint_pixels', 'albedo': .45}
    mean_rgb = rgb_sum / pixel_count
    ral = load_ral_table(ral_csv)
    row, _ = nearest_ral(mean_rgb, ral)
    return float(row['albedo']), {
        'mean_rgb': mean_rgb.tolist(),
        'ral': str(row['ral']),
        'albedo': float(row['albedo'])
    }


def run_fusion(images_dir,
               semantic_dir,
               bfms_dir,
               output_dir,
               ral_csv,
               paint_albedo=None):
    images = list_images(images_dir)
    output_dir = Path(output_dir)
    fused_dir = ensure_dir(output_dir / 'masks_fused')
    albedo_dir = ensure_dir(output_dir / 'albedo')
    if paint_albedo is None:
        paint_albedo, paint_meta = estimate_global_paint_albedo(
            images, semantic_dir, bfms_dir, ral_csv)
    else:
        paint_meta = None
    gate = build_gate_lut()
    alb_lut = build_albedo_lut(paint_albedo)
    rows = []
    material_groups = {
        'pct_Asphalt': ['Asphalt'],
        'pct_Cement_Concrete': ['Cement/Concrete'],
        'pct_Natural_Stone': ['Natural Stone'],
        'pct_Brick': ['Brick', 'Carved brick'],
        'pct_Soil_Mud': ['Soil/Mud'],
        'pct_Sand': ['Sand'],
        'pct_Roof_tile': ['Roof tile'],
        'pct_Glass': ['Glass'],
        'pct_Metal': ['Metal'],
        'pct_Paint': ['Paint/Coating/Plaster']
    }
    for i, p in enumerate(images, 1):
        sem = cv2.imread(str(Path(semantic_dir) / f'{p.stem}_sem.png'),
                         cv2.IMREAD_GRAYSCALE)
        bfms = cv2.imread(str(Path(bfms_dir) / f'{p.stem}_bfms.png'),
                          cv2.IMREAD_GRAYSCALE)
        if sem is None or bfms is None: continue
        valid = sem != IGNORE_INDEX
        fused = np.zeros_like(bfms)
        fused[valid] = gate[sem[valid], bfms[valid]]
        sem_idx = np.where(valid, sem, 0).astype(int)
        albedo = alb_lut[sem_idx, fused]
        albedo[~valid] = np.nan
        albedo[np.isin(sem, list(EXCLUDED_SEMANTIC))] = np.nan
        cv2.imwrite(str(fused_dir / f'{p.stem}_fused.png'), fused)
        np.save(albedo_dir / f'{p.stem}_albedo.npy', albedo.astype(np.float32))
        surface = valid & ~np.isin(sem, list(EXCLUDED_SEMANTIC))
        n = max(int(surface.sum()), 1)
        row = {
            'image':
            p.stem,
            'pixels_valides':
            int(valid.sum()),
            'pct_corriges_gating':
            float((valid & (fused != bfms)).sum() / max(valid.sum(), 1) * 100),
            'pixels_peinture_ral':
            int((fused == B['Paint/Coating/Plaster']).sum()),
            'couverture_albedo_pct':
            float(
                np.isfinite(albedo[surface]).mean() *
                100 if surface.any() else np.nan),
            'albedo_moyen':
            float(np.nanmean(albedo[surface]))
            if np.isfinite(albedo[surface]).any() else np.nan
        }
        for col, names in material_groups.items():
            row[col] = float(
                np.isin(fused, [B[x] for x in names])[surface].sum() / n * 100)
        for sid, name in [(0, 'road'), (1, 'sidewalk'), (2, 'building'),
                          (8, 'vegetation_haute'), (9, 'terrain_veg_basse')]:
            vals = albedo[(sem == sid) & np.isfinite(albedo)]
            row[f'albedo_{name}'] = float(vals.mean()) if vals.size else np.nan
        rows.append(row)
        if i % 25 == 0: print(f'[{i}] {p.name}')
    pd.DataFrame(rows).to_csv(output_dir / 'fusion_par_image.csv', index=False)
    if paint_meta:
        pd.DataFrame([paint_meta]).to_json(output_dir / 'paint_ral_match.json',
                                           orient='records',
                                           indent=2)
    return paint_albedo
