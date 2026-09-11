from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, pandas as pd
from coolpath.taxonomy import BFMS_NAME_TO_ID as B, SEM_ROAD, SEM_SIDEWALK, SEM_BUILDING, SEM_VEGETATION, SEM_TERRAIN, SEM_SKY, IGNORE_INDEX
from coolpath.indicators.geometry import angles_equirect, compute_svf, compute_gvi
from coolpath.utils.io import ensure_dir, list_images

IMPERMEABLE={B[x] for x in ['Asphalt','Cement/Concrete','Ground tile','Natural Stone','Engineered Stone/Imitation Stone','Brick','Carved brick','Roofing waterproof material']}
PERMEABLE={B[x] for x in ['Soil/Mud','Sand','Rammed earth']}
PERM_OVERRIDES={(SEM_TERRAIN,B['Foliage']):1,(SEM_TERRAIN,B['Tree']):1,(SEM_TERRAIN,B['Soil/Mud']):1,(SEM_TERRAIN,B['Sand']):1,(SEM_TERRAIN,B['Asphalt']):0,(SEM_TERRAIN,B['Cement/Concrete']):0,(SEM_TERRAIN,B['Ground tile']):0,(SEM_SIDEWALK,B['Foliage']):1,(SEM_ROAD,B['Foliage']):1}

def build_permeability_lut():
    lut=np.full((19,43),-1,dtype=np.int8)
    for m in IMPERMEABLE: lut[:,m]=0
    for m in PERMEABLE: lut[:,m]=1
    for key,val in PERM_OVERRIDES.items(): lut[key]=val
    return lut
PERM_LUT=build_permeability_lut()

def split_low_high_vegetation(sem: np.ndarray, elevation_threshold_deg=-7.0, azimuth_width_deg=12.0, seam_margin=60):
    veg=sem==SEM_VEGETATION; h,w=sem.shape; theta,_=angles_equirect(h,w); elev=90-np.degrees(theta)
    candidate=(veg & (elev[:,None]<elevation_threshold_deg)).astype(np.uint8)
    padded=np.pad(candidate,((0,0),(seam_margin,seam_margin)),mode='wrap')
    n,labels=cv2.connectedComponents(padded,connectivity=8); low=np.zeros_like(veg); deg_per_col=360.0/w
    for label_id in range(1,n):
        ys,xs=np.where(labels==label_id)
        if len(xs) and (xs.max()-xs.min()+1)*deg_per_col>=azimuth_width_deg:
            low[ys,(xs-seam_margin)%w]=True
    high=veg & ~low; return low,high

def compute_permeability(sem,fused,elevation_threshold_deg=-7.0,azimuth_width_deg=12.0):
    low,high=split_low_high_vegetation(sem,elevation_threshold_deg,azimuth_width_deg)
    valid=sem!=IGNORE_INDEX; soil=np.isin(sem,[SEM_ROAD,SEM_SIDEWALK,SEM_TERRAIN])&valid
    n_base=int(soil.sum()); n_low=int(low.sum()); n=n_base+n_low; canopy=float(high.sum()/max(valid.sum(),1)*100)
    if n==0: return np.nan,np.nan,np.nan,0,canopy
    classes=PERM_LUT[sem[soil].astype(int),fused[soil].astype(int)]
    n_perm=int((classes==1).sum())+n_low; n_imp=int((classes==0).sum()); n_unknown=int((classes==-1).sum())
    return 100*n_imp/n,100*n_perm/n,100*n_unknown/n,n,canopy

def compute_indicators_for_image(sem: np.ndarray, fused: np.ndarray, n_sectors=8, elevation_threshold_deg=-7.0, azimuth_width_deg=12.0):
    valid=sem!=IGNORE_INDEX; sky=sem==SEM_SKY; veg=sem==SEM_VEGETATION; terrain=sem==SEM_TERRAIN; building=sem==SEM_BUILDING
    svf=compute_svf(sky); gvi_high,gvi_high_omega=compute_gvi(veg,valid); gvi_low,_=compute_gvi(terrain,valid); gvi_total,_=compute_gvi(veg|terrain,valid); built,built_omega=compute_gvi(building,valid)
    imp,perm,unknown,nsoil,canopy=compute_permeability(sem,fused,elevation_threshold_deg,azimuth_width_deg)
    result={'SVF':svf,'GVI_haute_pct':gvi_high,'GVI_basse_pct':gvi_low,'GVI_total_pct':gvi_total,'GVI_haute_omega_pct':gvi_high_omega,'GVI_ecart_pt':gvi_high_omega-gvi_high,'taux_bati_pct':built,'taux_bati_omega_pct':built_omega,'taux_impermeable_pct':imp,'taux_permeable_pct':perm,'taux_indetermine_pct':unknown,'pixels_sol':nsoil,'canopee_sol_non_observable_pct':canopy}
    h,w=sem.shape
    for sec in range(n_sectors):
        a=int(sec*w/n_sectors); b=int((sec+1)*w/n_sectors); m=(veg|terrain)[:,a:b]; v=valid[:,a:b]
        result[f'GVI_sect{sec+1}_pct']=float(m[v].mean()*100) if v.any() else np.nan
    return result

def run_indicators(images_dir, semantic_dir, fused_dir, output_dir, elevation_threshold_deg=-7.0, azimuth_width_deg=12.0):
    output_dir=ensure_dir(output_dir); rows=[]
    for i,p in enumerate(list_images(images_dir),1):
        sem=cv2.imread(str(Path(semantic_dir)/f'{p.stem}_sem.png'),cv2.IMREAD_GRAYSCALE); fused=cv2.imread(str(Path(fused_dir)/f'{p.stem}_fused.png'),cv2.IMREAD_GRAYSCALE)
        if sem is None or fused is None: continue
        h,w=sem.shape
        if abs(w/h-2.0)>=.05: raise ValueError(f'{p.name}: ratio {w/h:.3f} non équirectangulaire')
        row={'image':p.stem}; row.update(compute_indicators_for_image(sem,fused,elevation_threshold_deg=elevation_threshold_deg,azimuth_width_deg=azimuth_width_deg)); rows.append(row)
        if i%25==0: print(f'[{i}] {p.name}')
    df=pd.DataFrame(rows); df.to_csv(Path(output_dir)/'indicateurs_par_image.csv',index=False); return df
