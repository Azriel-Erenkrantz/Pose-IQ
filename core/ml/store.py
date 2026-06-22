"""
Step 5 (PI-86): Model store — single source of truth for where trained
artifacts live and how they are (de)serialised.

Both the trainer (writes) and the predictor (reads) go through here so the
on-disk layout is defined in exactly one place, mirroring how `build_row` is
the one schema shared by the live collector and the offline extractor.

Layout:
    data/models/<head>.joblib     one bundle per head (quality / phase / exercise)
    data/models/registry.json     human-readable summary of every saved bundle

A "bundle" is a dict:
    {
        'name':          str,                # head name, e.g. 'quality'
        'kind':          str,                # 'classifier' (good/bad, phase, exercise)
                                             # or 'oneclass' (anomaly: trained on good only)
        'pipeline':      sklearn estimator,  # fitted, takes a (n, N_FEATURES) matrix
        'classes':       list,               # class labels (classifier heads); None for oneclass
        'feature_names': list[str],          # == features.FEATURE_NAMES at train time
        'window':        int,                # window length the features assume
        'stride':        int,
        'trained_at':    str,                # ISO timestamp
        'n_samples':     int,
        'metrics':       dict,               # accuracy / per-class report / inlier rate
    }
"""
import json
import os
from datetime import datetime, timezone

import joblib

from ml.data_collector import DATA_DIR

MODELS_DIR = os.path.join(DATA_DIR, 'models')
REGISTRY = os.path.join(MODELS_DIR, 'registry.json')


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f'{name}.joblib')


def save_model(name: str, pipeline, classes, feature_names, window, stride,
               n_samples, metrics, kind: str = 'classifier') -> str:
    """Persist one fitted head as a bundle and refresh the registry. Returns path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        'name': name,
        'kind': kind,
        'pipeline': pipeline,
        'classes': list(classes) if classes is not None else None,
        'feature_names': list(feature_names),
        'window': window,
        'stride': stride,
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_samples': int(n_samples),
        'metrics': metrics,
    }
    path = model_path(name)
    joblib.dump(bundle, path)
    _update_registry(bundle)
    return path


def load_model(name: str) -> dict | None:
    """Load a saved bundle, or None if this head has not been trained yet."""
    path = model_path(name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def list_models() -> list:
    """Names of every head that currently has a saved bundle on disk."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(MODELS_DIR)
        if f.endswith('.joblib')
    )


def _update_registry(bundle: dict):
    """Keep a small JSON index of saved bundles (everything except the estimator)."""
    registry = {}
    if os.path.exists(REGISTRY):
        try:
            with open(REGISTRY, encoding='utf-8') as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            registry = {}

    registry[bundle['name']] = {
        k: bundle[k] for k in
        ('kind', 'classes', 'window', 'stride', 'trained_at', 'n_samples', 'metrics')
    }
    with open(REGISTRY, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
