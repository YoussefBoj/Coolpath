"""YAML loading. Relative paths are anchored to project_root, never to src/."""
from __future__ import annotations
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.name == 'src':
    REPO_ROOT = REPO_ROOT.parent


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            'Le fichier YAML doit contenir des paramètres nommés.')
    root = Path(
        cfg.get('project_root',
                '..' if path.parent.name == 'configs' else '.'))
    cfg['_root'] = str((path.parent / root).resolve())
    cfg['_config_file'] = str(path)
    return cfg


def repo_path(value: str | Path | None,
              cfg: dict | None = None) -> Path | None:
    if value is None or str(value).strip() == '':
        return None
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(
        (cfg or {}).get('_root', REPO_ROOT)) / p).resolve()


def get_path(cfg: dict, key: str) -> Path:
    cur = cfg
    for part in key.split('.'):
        cur = cur[part]
    path = repo_path(cur, cfg)
    if path is None:
        raise ValueError(f'Chemin manquant : {key}')
    return path
