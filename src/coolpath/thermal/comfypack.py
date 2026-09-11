from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

COLMAP={
'TIMESTAMP (TS)':'ts','TIMESTAMP':'ts','Prt_Ball_Temperature_Avg (Avg)':'Tg','Prt_Ball_Temperature_Avg':'Tg','Prt_Mid_Temperature_Avg (degC) (Avg)':'Prt_Mid','Prt_Mid_Temperature_Avg (Avg)':'Prt_Mid','Prt_Mid_Temperature_Avg':'Prt_Mid','HMP_Temp (Smp)':'HMP_Temp','HMP_Temp':'HMP_Temp','HMP_RH (Smp)':'RH','HMP_RH':'RH','WS_avg (m/s) (WVc)':'WS','WS_avg':'WS','WindDir_vct (Deg) (WVc)':'WD','WindDir_vct':'WD','Solspy (Smp)':'Kdown','Solspy':'Kdown','Evenement (Smp)':'event','Evenement':'event','Inclinaison_X (Smp)':'inc_x','Inclinaison_X':'inc_x','Inclinaison_Y (Smp)':'inc_y','Inclinaison_Y':'inc_y'}
NUM_COLS=['Tg','Prt_Mid','HMP_Temp','RH','WS','WD','Kdown','inc_x','inc_y']
RANGES={'Ta':(-20,55),'Tg':(-20,90),'Prt_Mid':(-20,55),'RH':(0,100),'WS':(0,40),'WD':(0,360),'Kdown':(0,1400),'inc_x':(-90,90),'inc_y':(-90,90)}

def load_comfypack(path: str | Path, ta_source='HMP_Temp') -> pd.DataFrame:
    path=Path(path); first=path.open(encoding='utf-8',errors='replace').readline()
    if first.startswith('"TOA5"') or first.startswith('TOA5') or first.count(';')>3:
        df=pd.read_csv(path,sep=';',skiprows=[0,3,4],header=1,encoding='utf-8'); df.columns=[c.strip().strip('"') for c in df.columns]
    else: df=pd.read_csv(path)
    df=df.rename(columns={k:v for k,v in COLMAP.items() if k in df.columns})
    if 'ts' not in df: raise KeyError('Colonne horodatage introuvable')
    df['ts']=pd.to_datetime(df['ts'],format='mixed')
    for c in NUM_COLS:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['Ta']=df[ta_source if ta_source in df.columns else 'HMP_Temp']
    return df.sort_values('ts').reset_index(drop=True)

def despike_mad(series: pd.Series, window=31, n_mad=5.0):
    med=series.rolling(window,center=True,min_periods=5).median(); mad=(series-med).abs().rolling(window,center=True,min_periods=5).median(); threshold=n_mad*1.4826*mad
    bad=(series-med).abs()>threshold.replace(0,np.nan); return series.mask(bad),int(bad.sum())

def quality_control(df: pd.DataFrame, gradient_max_k_s=.5) -> tuple[pd.DataFrame,pd.DataFrame]:
    df=df.copy(); report=[]; dt=df.ts.diff().dt.total_seconds()
    for col,(lo,hi) in RANGES.items():
        if col not in df: continue
        n0=int(df[col].isna().sum()); df[col]=df[col].where(df[col].between(lo,hi)); n_range=int(df[col].isna().sum()-n0); n_grad=n_mad=0
        if col in ['Ta','Tg','Prt_Mid']:
            grad=df[col].diff().abs()/dt; bad=grad>gradient_max_k_s; n_grad=int(bad.sum()); df.loc[bad,col]=np.nan; df[col],n_mad=despike_mad(df[col])
        report.append({'variable':col,'hors_plage':n_range,'gradient':n_grad,'mad':n_mad})
    for c in ['Ta','Tg','RH','WS']:
        if c in df: df[c]=df[c].interpolate(limit=5,limit_direction='both')
    return df,pd.DataFrame(report)
