from __future__ import annotations
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import cv2
import numpy as np
from coolpath.taxonomy import CITYSCAPES_CLASSES, CITYSCAPES_PALETTE
from coolpath.utils.io import ensure_dir

def segment_equirect_cubemap(model, equi_bgr: np.ndarray, face_width: int = 768) -> np.ndarray:
    import py360convert
    from mmseg.apis import inference_model
    h, w = equi_bgr.shape[:2]
    rgb = cv2.cvtColor(equi_bgr, cv2.COLOR_BGR2RGB)
    faces = py360convert.e2c(rgb, face_w=face_width, cube_format='dict')
    pred_faces = {}
    for key, face_rgb in faces.items():
        face_bgr = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = inference_model(model, face_bgr)
        pred = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        pred_faces[key] = pred[..., None].astype(np.float32)
    equi = py360convert.c2e(pred_faces, h=h, w=w, mode='nearest', cube_format='dict')
    return np.rint(equi[..., 0]).astype(np.uint8)

def export_cvat_polygons(model, image_paths: list[Path], xml_path: str | Path,
                         preview_dir: str | Path | None = None,
                         face_width: int = 768, min_contour_area: int = 600) -> Path:
    root = ET.Element('annotations'); ET.SubElement(root,'version').text='1.1'
    labels_el = ET.SubElement(ET.SubElement(ET.SubElement(root,'meta'),'task'),'labels')
    for name, color in zip(CITYSCAPES_CLASSES, CITYSCAPES_PALETTE):
        le=ET.SubElement(labels_el,'label'); ET.SubElement(le,'name').text=name
        ET.SubElement(le,'color').text='#%02x%02x%02x' % tuple(color)
        ET.SubElement(le,'type').text='polygon'; ET.SubElement(le,'attributes')
    preview_dir = ensure_dir(preview_dir) if preview_dir else None
    for image_id, path in enumerate(image_paths):
        orig=cv2.imread(str(path)); h,w=orig.shape[:2]
        mask=segment_equirect_cubemap(model,orig,face_width)
        if preview_dir:
            color_mask=np.zeros((h,w,3),dtype=np.uint8)
            for cid in np.unique(mask):
                if cid < len(CITYSCAPES_CLASSES): color_mask[mask==cid]=CITYSCAPES_PALETTE[cid][::-1]
            cv2.imwrite(str(preview_dir/path.name),cv2.addWeighted(orig,.6,color_mask,.4,0))
        image_el=ET.SubElement(root,'image',id=str(image_id),name=path.name,width=str(w),height=str(h))
        for cid in np.unique(mask):
            if cid >= len(CITYSCAPES_CLASSES): continue
            contours,_=cv2.findContours((mask==cid).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt)<min_contour_area: continue
                approx=cv2.approxPolyDP(cnt,0.002*cv2.arcLength(cnt,True),True)
                if len(approx)<3: continue
                pts=';'.join(f'{p[0][0]:.1f},{p[0][1]:.1f}' for p in approx)
                ET.SubElement(image_el,'polygon',label=CITYSCAPES_CLASSES[int(cid)],source='auto',occluded='0',points=pts,z_order='0')
    xml_path=Path(xml_path); ensure_dir(xml_path.parent)
    xml_path.write_text(minidom.parseString(ET.tostring(root)).toprettyxml(indent='  '),encoding='utf-8')
    return xml_path

def write_cvat_labels(path: str | Path) -> Path:
    payload=[]
    for name,color in zip(CITYSCAPES_CLASSES,CITYSCAPES_PALETTE):
        payload.append({'name':name,'color':'#%02x%02x%02x'%tuple(color),'type':'polygon','attributes':[]})
    path=Path(path); ensure_dir(path.parent); path.write_text(json.dumps(payload,indent=2),encoding='utf-8'); return path
