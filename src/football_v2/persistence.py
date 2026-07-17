from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .model import TailAwareExactScoreModel


@dataclass(frozen=True)
class ModelBundle:
    model: TailAwareExactScoreModel
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)
    return target


def load_bundle(path: str | Path) -> ModelBundle:
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, ModelBundle):
        raise TypeError("artifact does not contain a ModelBundle")
    return loaded
