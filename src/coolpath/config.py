from __future__ import annotations
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg['_config_file'] = str(path.resolve())
    return cfg

def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()

def get_path(cfg: dict, key: str) -> Path:
    cur = cfg
    for part in key.split('.'):
        cur = cur[part]
    return repo_path(cur)
