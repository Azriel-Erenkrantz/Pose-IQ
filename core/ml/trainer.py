"""
Step 5 (PI-84/85): Training pipeline.

Reads the collected frames (live `training_data.csv` + offline `video_data.csv`),
turns them into windowed feature vectors via `features.extract_windows`, then
fits one independent classifier per "head":

    quality  : good / bad form        (only from clip-labelled video rows)
    phase     : which phase of the rep (all rows)
    exercise  : which exercise         (all rows)

Each head is trained, evaluated on a held-out split, and saved through
`ml.store`. Heads with too little data (or a single class) are skipped with a
message instead of crashing — so this runs cleanly today, before the videos are
in, and again unchanged once they are.

Run:  python core/ml/trainer.py
"""
import os
import sys

# Runnable as a plain script: put the package root (core/) on the path first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ml.data_collector import TRAINING_CSV, HEADER
from ml.features import extract_windows, FEATURE_NAMES, DEFAULT_WINDOW, DEFAULT_STRIDE
from ml.video_extractor import VIDEO_CSV
from ml import store

# Don't try to fit a head until there is at least this much to learn from.
MIN_WINDOWS = 30
MIN_PER_CLASS = 5
TEST_SIZE = 0.25
RANDOM_STATE = 42

# head name -> (label column in y, set of label values to keep or None for all)
HEADS = {
    'quality':  ('quality_label', {'good', 'bad'}),   # clip-labelled video rows only
    'phase':    ('phase_name', None),
    'exercise': ('exercise_id', None),
}


def load_frames(csv_paths=None) -> pd.DataFrame:
    """Concatenate every available source CSV into one frame table (shared schema)."""
    csv_paths = csv_paths or [TRAINING_CSV, VIDEO_CSV]
    frames = []
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
            print(f"  loaded {len(df):>6} rows from {os.path.basename(path)}")
    if not frames:
        return pd.DataFrame(columns=HEADER)
    return pd.concat(frames, ignore_index=True)


def _build_estimator():
    """The classifier used for every head. RandomForest: robust on small tabular data,
    needs no feature scaling, gives class probabilities for live confidence."""
    from sklearn.ensemble import RandomForestClassifier
    # TODO(data): tune n_estimators / max_depth once real video volume is in.
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_head(name: str, X, y, label_col, keep_labels,
               window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE):
    """Fit, evaluate and save one head. Returns metrics dict, or None if skipped."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    if label_col not in y.columns:
        print(f"  [{name}] skip — no '{label_col}' column in the data.")
        return None

    mask = y[label_col].notna()
    if keep_labels is not None:
        mask &= y[label_col].isin(keep_labels)
    Xf, yf = X[mask.to_numpy()], y.loc[mask, label_col].astype(str)

    if len(Xf) < MIN_WINDOWS:
        print(f"  [{name}] skip — only {len(Xf)} windows (need {MIN_WINDOWS}).")
        return None

    counts = yf.value_counts()
    classes = counts[counts >= MIN_PER_CLASS].index.tolist()
    if len(classes) < 2:
        have = dict(counts)
        print(f"  [{name}] skip — need >=2 classes with >={MIN_PER_CLASS} windows, have {have}.")
        return None

    keep = yf.isin(classes)
    Xf, yf = Xf[keep.to_numpy()], yf[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xf, yf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yf
    )

    clf = _build_estimator()
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    metrics = {
        'accuracy': round(float(accuracy_score(y_te, y_pred)), 4),
        'report': classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        'n_train': len(X_tr),
        'n_test': len(X_te),
    }

    path = store.save_model(
        name=name, pipeline=clf, classes=clf.classes_,
        feature_names=FEATURE_NAMES, window=window, stride=stride,
        n_samples=len(Xf), metrics=metrics,
    )
    print(f"  [{name}] trained on {len(Xf)} windows, "
          f"acc={metrics['accuracy']:.3f}, classes={list(clf.classes_)} -> {os.path.basename(path)}")
    return metrics


def train_quality_oneclass(X, y, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE):
    """Fallback when there are no 'bad' clips: learn 'good' form as anomaly detection.

    One IsolationForest per exercise, fitted on that exercise's 'good' windows. At
    inference a live window that falls OUTSIDE the learned good-form envelope is
    flagged as off-form. Saved as head 'quality_<exercise_id>' (kind='oneclass').
    Per-exercise because good squat form and good lunge form occupy different
    regions of feature space; the live loop always knows which exercise is active.
    """
    from sklearn.ensemble import IsolationForest

    if 'quality_label' not in y.columns:
        print("  [quality_oneclass] skip — no 'quality_label' column.")
        return None

    good = y['quality_label'].astype(str).eq('good')
    if not good.any():
        print("  [quality_oneclass] skip — no 'good'-labelled windows.")
        return None

    Xg = X[good.to_numpy()]
    yg = y[good].reset_index(drop=True)

    trained = []
    for ex_id, positions in yg.groupby('exercise_id').groups.items():
        Xe = Xg[list(positions)]
        if len(Xe) < MIN_WINDOWS:
            print(f"  [quality_{ex_id}] skip — only {len(Xe)} good windows (need {MIN_WINDOWS}).")
            continue
        # TODO(data): set contamination from a small held-out 'bad' set once available.
        clf = IsolationForest(n_estimators=200, contamination='auto',
                              random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(Xe)
        inlier_rate = round(float((clf.predict(Xe) == 1).mean()), 4)
        metrics = {'inlier_rate_train': inlier_rate, 'n_good': len(Xe)}
        store.save_model(name=f'quality_{ex_id}', pipeline=clf, classes=None,
                         feature_names=FEATURE_NAMES, window=window, stride=stride,
                         n_samples=len(Xe), metrics=metrics, kind='oneclass')
        print(f"  [quality_{ex_id}] one-class on {len(Xe)} good windows, "
              f"train inlier-rate={inlier_rate:.2f} -> quality_{ex_id}.joblib")
        trained.append(ex_id)

    return trained or None


def train_all(window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE):
    print("Loading frames...")
    df = load_frames()
    if df.empty:
        print("No data yet. Run a workout (live) or `python core/ml/video_extractor.py` first.")
        return {}

    print(f"Building windows (window={window}, stride={stride})...")
    X, y = extract_windows(df, window=window, stride=stride)
    print(f"  -> {X.shape[0]} windows x {X.shape[1]} features\n")
    if X.shape[0] == 0:
        print("No complete windows — sessions are shorter than one window.")
        return {}

    results = {}
    for name, (label_col, keep_labels) in HEADS.items():
        results[name] = train_head(name, X, y, label_col, keep_labels, window, stride)

    # No 'bad' clips yet -> fall back to per-exercise one-class on the 'good' windows,
    # so good-only data still produces a form signal.
    if not results.get('quality'):
        results['quality_oneclass'] = train_quality_oneclass(X, y, window, stride)

    trained = [n for n, m in results.items() if m]
    print(f"\nDone. Trained: {trained or 'none'}.  Models in {store.MODELS_DIR}")
    return results


if __name__ == '__main__':
    train_all()
