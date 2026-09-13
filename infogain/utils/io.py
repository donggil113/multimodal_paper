"""Small IO helpers: JSON with numpy support, YAML configs, run directories."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):  # noqa: D102
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, cls=NumpyJSONEncoder)
    return path


def load_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def load_yaml(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_dir(root: str | Path, name: str, timestamp: bool = False) -> Path:
    root = Path(root)
    out = root / (f"{name}_{datetime.now():%Y%m%d-%H%M%S}" if timestamp else name)
    out.mkdir(parents=True, exist_ok=True)
    return out
