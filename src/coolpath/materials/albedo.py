from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from skimage.color import rgb2lab
from coolpath.taxonomy import BFMS_NAME_TO_ID, SEM_VEGETATION, SEM_TERRAIN
B=BFMS_NAME_TO_ID

BASE_ALBEDO = {
 B['Asphalt']:.10, B['Cement/Concrete']:.29, B['Natural Stone']:.29,
 B['Brick']:.24, B['Carved brick']:.24, B['Soil/Mud']:.18, B['Sand']:.40,
 B['Roof tile']:.35, B['Glass']:.12, B['Metal']:.38, B['Paint/Coating/Plaster']:.45,
 B['Foliage']:.22, B['Tree']:.22,
}
ALBEDO_OVERRIDES={(SEM_VEGETATION,B['Foliage']):.20,(SEM_VEGETATION,B['Tree']):.20,(SEM_TERRAIN,B['Foliage']):.21,(SEM_TERRAIN,B['Tree']):.21,(SEM_TERRAIN,B['Soil/Mud']):.17,(SEM_TERRAIN,B['Sand']):.40}
EXCLUDED_SEMANTIC={10,11,12}

def build_albedo_lut(paint_albedo: float | None=None) -> np.ndarray:
    lut=np.full((19,43),np.nan,dtype=float)
    for mat,val in BASE_ALBEDO.items(): lut[:,mat]=val
    for key,val in ALBEDO_OVERRIDES.items(): lut[key]=val
    if paint_albedo is not None: lut[:,B['Paint/Coating/Plaster']]=paint_albedo
    return lut

def load_ral_table(csv_path: str | Path) -> pd.DataFrame:
    df=pd.read_csv(csv_path); return df

def nearest_ral(rgb: np.ndarray, ral_df: pd.DataFrame) -> tuple[pd.Series,float]:
    rgb=np.asarray(rgb,dtype=float).reshape(1,1,3)/255.0
    target=rgb2lab(rgb)[0,0]
    refs=rgb2lab(ral_df[['R','G','B']].to_numpy(dtype=float).reshape(1,-1,3)/255.0)[0]
    idx=int(np.argmin(np.linalg.norm(refs-target,axis=1)))
    return ral_df.iloc[idx], float(np.linalg.norm(refs[idx]-target))

def shadow_correct_rgb(rgb_pixels: np.ndarray, percentile: float=95) -> np.ndarray:
    rgb=np.asarray(rgb_pixels,dtype=float)
    lum=.299*rgb[:,0]+.587*rgb[:,1]+.114*rgb[:,2]
    ref=float(np.percentile(lum,percentile)); scale=ref/np.maximum(lum,1.0)
    return np.clip(rgb*scale[:,None],0,255)
