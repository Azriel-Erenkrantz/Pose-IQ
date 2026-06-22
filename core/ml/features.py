"""
PI-83/129: Feature engineering for the ML layer.

Turns a window of ~20-30 per-frame angle snapshots into a fixed-length feature
vector. The SAME function is used offline (training, over the collected CSV) and
online (inference, over a rolling buffer of live frames) so the two never drift.

Per joint we compute 5 aggregates -> len(ANGLE_COLS) * 5 features.
Missing joints are stored as -1.0 in the CSV; we treat that as NaN, aggregate
nan-aware, and refill fully-absent joints with a MISSING sentinel so the model
can learn "this joint was not visible".
"""
import os
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ml.data_collector import ANGLE_COLS, TRAINING_CSV

AGG_FUNCS = ['mean', 'min', 'max', 'std', 'range']
FEATURE_NAMES = [f'{joint}_{agg}' for joint in ANGLE_COLS for agg in AGG_FUNCS]
N_FEATURES = len(FEATURE_NAMES)        # 8 joints * 5 = 40

MISSING = -1.0                         # sentinel for an unseen joint (angles are 0..180)
DEFAULT_WINDOW = 25
DEFAULT_STRIDE = 10


def window_features(window: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Aggregate one window into a fixed 40-dim feature vector.

    Args:
        window: shape (n_frames, len(ANGLE_COLS)). Values of MISSING (-1) mean
                the joint was not visible in that frame.
    Returns:
        1-D array of length N_FEATURES, ordered to match FEATURE_NAMES.
    """
    arr = np.asarray(window, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(ANGLE_COLS):
        raise ValueError(f"window must be (n_frames, {len(ANGLE_COLS)}), got {arr.shape}")

    arr = np.where(arr == MISSING, np.nan, arr)      # treat -1 as missing

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)   # all-NaN columns
        means = np.nanmean(arr, axis=0)
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        stds = np.nanstd(arr, axis=0)

    ranges = maxs - mins
    # Stack as (n_joints, 5) in AGG_FUNCS order, then flatten to match FEATURE_NAMES.
    feats = np.stack([means, mins, maxs, stds, ranges], axis=1).reshape(-1)
    return np.nan_to_num(feats, nan=MISSING)


def features_from_dicts(frames: Sequence[Dict[str, float]]) -> np.ndarray:
    """Runtime helper: list of {joint: angle} dicts -> 40-dim vector (for the pipeline buffer)."""
    window = [[f.get(joint, MISSING) for joint in ANGLE_COLS] for f in frames]
    return window_features(window)


def extract_windows(
    df,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
) -> Tuple[np.ndarray, "pd.DataFrame"]:
    """
    Slide a window over each session's frames and build the training matrix.

    Windows never cross a session_id boundary. A window's labels are derived from
    its rows: exercise = the (single) exercise, phase = the most common phase,
    quality = mean quality_score, in_phase = fraction of clean frames.

    Returns:
        X: (n_windows, N_FEATURES) float array
        y: DataFrame with columns [exercise_id, phase_name, quality, in_phase_ratio]
    """
    import pandas as pd

    X_rows: List[np.ndarray] = []
    y_rows: List[dict] = []

    for session_id, g in df.groupby('session_id', sort=False):
        g = g.reset_index(drop=True)
        angles = g[ANGLE_COLS].to_numpy(dtype=float)
        n = len(g)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            sl = slice(start, start + window)
            X_rows.append(window_features(angles[sl]))
            chunk = g.iloc[sl]
            row = {
                'exercise_id': chunk['exercise_id'].mode().iat[0],
                'phase_name': chunk['phase_name'].mode().iat[0],
                'quality': round(float(chunk['quality_score'].mean()), 2),
                'in_phase_ratio': round(float(chunk['in_phase'].mean()), 3),
            }
            # clip-level good/bad label, present only in video-sourced rows
            if 'quality_label' in chunk:
                row['quality_label'] = chunk['quality_label'].mode().iat[0]
            y_rows.append(row)

    X = np.array(X_rows, dtype=float) if X_rows else np.empty((0, N_FEATURES))
    y = pd.DataFrame(y_rows)
    return X, y


def build_dataset(
    csv_path: str = None,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    out_path: str = None,
):
    """Load the collected CSV, build windowed features, optionally save to .npz."""
    import pandas as pd

    csv_path = csv_path or TRAINING_CSV
    df = pd.read_csv(csv_path)
    X, y = extract_windows(df, window=window, stride=stride)

    if out_path:
        np.savez_compressed(
            out_path,
            X=X,
            feature_names=np.array(FEATURE_NAMES),
            **{col: y[col].to_numpy() for col in y.columns},
        )
    return X, y


if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else TRAINING_CSV
    if not os.path.exists(path):
        print(f"No data at {path}. Run a workout first so the collector writes rows.")
        sys.exit(0)

    X, y = build_dataset(path)
    print(f"windows: {X.shape[0]}   features/window: {X.shape[1]}")
    if len(y):
        print("exercises:", dict(y['exercise_id'].value_counts()))
        print("phases:   ", dict(y['phase_name'].value_counts()))
        print("quality:   mean={:.1f}  min={:.1f}  max={:.1f}".format(
            y['quality'].mean(), y['quality'].min(), y['quality'].max()))
