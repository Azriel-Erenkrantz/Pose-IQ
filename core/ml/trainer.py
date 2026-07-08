"""
Train a per-exercise phase classifier from hand-labeled videos.

Pipeline:
  1. Read label JSON files from data/test_labels/{exercise_id}/*.json
  2. For each labeled frame extract joint angles + rolling angle delta.
  3. Build a 13-feature vector: 12 joint angles (-1 if missing) + 1 delta.
  4. Hold out the last label file (alphabetically) per exercise for evaluation.
  5. Train a Random Forest on the remaining files, evaluate on the held-out file.
  6. Save model to data/models/phase_{exercise_id}.joblib.

Label file format:
  {
    "exercise_id": "squat",
    "video": "data/videos/squat/my_video.mp4",
    "skip": 1,
    "labels": [
      {"start_sec": 0.0,  "end_sec": 2.5,  "phase": "standing"},
      {"start_sec": 2.5,  "end_sec": 5.0,  "phase": "descending"},
      {"start_sec": 5.0,  "end_sec": 7.2,  "phase": "ascending"}
    ]
  }

Usage:
    python -m core.ml.trainer                        # all exercises
    python -m core.ml.trainer --exercise squat       # one exercise
    python -m core.ml.trainer --labels data/my_labels --quiet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from core.exercise.exercise_model import ExerciseModel

JOINTS = [
    'right_knee',    'left_knee',
    'right_hip',     'left_hip',
    'right_ankle',   'left_ankle',
    'right_elbow',   'left_elbow',
    'right_shoulder','left_shoulder',
    'right_wrist',   'left_wrist',
]

MODELS_DIR  = Path(__file__).parent.parent.parent / 'data' / 'models'
LABELS_DIR  = Path(__file__).parent.parent.parent / 'data' / 'test_labels'
PROJECT_ROOT = Path(__file__).parent.parent.parent

MIN_SAMPLES_PER_CLASS = 10
DELTA_WINDOW = 5


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def angles_to_features(angles: Dict[str, float], angle_delta: float = 0.0) -> np.ndarray:
    """12 joint angles + 1 rolling delta = 13 features. Missing joints → -1."""
    return np.array([angles.get(j, -1.0) for j in JOINTS] + [angle_delta], dtype=np.float32)


# ---------------------------------------------------------------------------
# Label-file helpers (shared logic with eval.py)
# ---------------------------------------------------------------------------

def _video_fps(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def _resolve_video(video_str: str, label_file: Path) -> Optional[Path]:
    p = Path(video_str)
    if p.is_absolute() and p.exists():
        return p
    for base in (label_file.parent, PROJECT_ROOT):
        c = base / p
        if c.exists():
            return c
    return None


def _build_frame_labels(labels: List[dict], n_frames: int, fps: float, skip: int) -> Dict[int, str]:
    frame_labels: Dict[int, str] = {}
    for lbl in labels:
        phase = lbl['phase']
        if 'start_sec' in lbl:
            start_idx = int(lbl['start_sec'] * fps / skip)
            end_idx   = int(lbl['end_sec']   * fps / skip)
        else:
            start_idx = int(lbl['start_frame'] / skip)
            end_idx   = int(lbl['end_frame']   / skip)
        for i in range(start_idx, min(end_idx, n_frames)):
            frame_labels[i] = phase
    return frame_labels


def _rolling_delta(per_frame: List[Dict[str, float]], frame_idx: int,
                   primary: Optional[str], window: int = DELTA_WINDOW) -> float:
    if primary is None or frame_idx < 1:
        return 0.0
    deltas = []
    for i in range(max(1, frame_idx - window + 1), frame_idx + 1):
        prev = per_frame[i - 1].get(primary)
        curr = per_frame[i].get(primary)
        if prev is not None and curr is not None:
            deltas.append(curr - prev)
    return sum(deltas) / len(deltas) if deltas else 0.0


def _primary_joint(per_frame: List[Dict[str, float]], labeled_indices: List[int]) -> Optional[str]:
    """Joint with highest variance across labeled frames — most informative signal."""
    variances: Dict[str, float] = {}
    for j in JOINTS:
        vals = [per_frame[i][j] for i in labeled_indices if j in per_frame[i]]
        if len(vals) > 1:
            variances[j] = float(np.var(vals))
    return max(variances, key=variances.get) if variances else None


# ---------------------------------------------------------------------------
# Sample collection from hand-labeled files
# ---------------------------------------------------------------------------

def collect_samples_from_labels(
    exercise_id: str,
    labels_root: Path = LABELS_DIR,
    skip: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], Optional[Path], Dict[str, Dict[str, List[float]]]]:
    """
    Build (X, y) training data from hand-labeled JSON files.

    Also accumulates raw angle values per phase/joint across all training files
    so the caller can compute min/max/mean/std and write to exercise_angles.

    Holds out the last file alphabetically as the eval set.
    Returns (X, y, eval_label_path, phase_angles).
    """
    from collections import defaultdict
    from core.ml.angles import compute_angles
    from core.ml.extractor import extract_frames

    label_dir = labels_root / exercise_id
    if not label_dir.exists():
        if verbose:
            print(f'  [{exercise_id}] no label directory: {label_dir}')
        return np.empty((0, len(JOINTS) + 1)), [], None, {}

    label_files = sorted(label_dir.glob('*.json'))
    if not label_files:
        if verbose:
            print(f'  [{exercise_id}] no .json label files found')
        return np.empty((0, len(JOINTS) + 1)), [], None, {}

    if len(label_files) == 1:
        if verbose:
            print(f'  [{exercise_id}] only 1 label file — using for training, no eval holdout')
        train_files, eval_file = label_files, None
    else:
        eval_file   = label_files[-1]
        train_files = label_files[:-1]
        if verbose:
            print(f'  [{exercise_id}] train={len(train_files)} files  eval={eval_file.name}')

    X_rows, y_rows = [], []
    # phase → joint → [angle values]  accumulated across all training videos
    phase_angles: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for lf in train_files:
        data = json.loads(lf.read_text(encoding='utf-8'))
        video_path = _resolve_video(data['video'], lf)
        if video_path is None:
            if verbose:
                print(f'    {lf.name}: video not found, skipping')
            continue

        file_skip = data.get('skip', skip)
        if verbose:
            print(f'    {lf.name} ...', end=' ', flush=True)

        try:
            frames = extract_frames(video_path, skip=file_skip)
            if len(frames) < 5:
                if verbose:
                    print('too short, skipped')
                continue

            per_frame = [compute_angles(f) for f in frames]
            fps       = _video_fps(video_path)
            frame_labels = _build_frame_labels(data['labels'], len(frames), fps, file_skip)

            if not frame_labels:
                if verbose:
                    print('no labeled frames — check timestamps')
                continue

            primary = _primary_joint(per_frame, list(frame_labels.keys()))

            labeled = 0
            for i, angles in enumerate(per_frame):
                if i not in frame_labels:
                    continue
                phase = frame_labels[i]
                delta = _rolling_delta(per_frame, i, primary)
                X_rows.append(angles_to_features(angles, delta))
                y_rows.append(phase)
                for joint, value in angles.items():
                    phase_angles[phase][joint].append(value)
                labeled += 1

            if verbose:
                phases = list(dict.fromkeys(frame_labels.values()))
                print(f'{len(frames)} frames → {labeled} labeled  {phases}')

        except Exception as exc:
            if verbose:
                print(f'ERROR: {exc}')

    if not X_rows:
        return np.empty((0, len(JOINTS) + 1)), [], eval_file, dict(phase_angles)

    return np.vstack(X_rows), y_rows, eval_file, dict(phase_angles)


def write_exercise_angles(
    exercise_id: str,
    phase_angles: Dict[str, Dict[str, List[float]]],
    verbose: bool = True,
) -> None:
    """
    Compute min/max/mean/std per phase/joint from collected training frames
    and write to the exercise_angles MongoDB collection.

    This replaces the Model 2 builder step — the same labeled frames that
    train the RF also populate the form-correction angle database.
    """
    from core.db import get_db
    db = get_db()

    for phase, joints in phase_angles.items():
        joint_stats = {}
        for joint, values in joints.items():
            if not values:
                continue
            arr = np.array(values)
            joint_stats[joint] = {
                'min':      round(float(arr.min()), 1),
                'max':      round(float(arr.max()), 1),
                'mean':     round(float(arr.mean()), 1),
                'std':      round(float(arr.std()), 1),
                'n_frames': len(values),
            }

        db.exercise_angles.replace_one(
            {'exercise_id': exercise_id, 'phase': phase},
            {'exercise_id': exercise_id, 'phase': phase, 'joints': joint_stats},
            upsert=True,
        )

    if verbose:
        phases = sorted(phase_angles.keys())
        print(f'  [{exercise_id}] exercise_angles updated: {phases}')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_exercise(
    exercise_id: str,
    X: np.ndarray,
    y: List[str],
    verbose: bool = True,
) -> Optional[Path]:
    """Fit a RandomForest for one exercise and save it. Returns save path or None."""
    from collections import Counter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if len(X) == 0:
        if verbose:
            print(f'  [{exercise_id}] no samples, skipping')
        return None

    counts = Counter(y)
    valid_classes = {cls for cls, n in counts.items() if n >= MIN_SAMPLES_PER_CLASS}
    if len(valid_classes) < 2:
        if verbose:
            print(f'  [{exercise_id}] need ≥2 classes with ≥{MIN_SAMPLES_PER_CLASS} samples. '
                  f'Have: {dict(counts)}')
        return None

    mask = np.array([label in valid_classes for label in y])
    X_f  = X[mask]
    y_f  = np.array(y)[mask]

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    cv_acc = None
    if len(X_f) >= 20:
        scores = cross_val_score(clf, X_f, y_f, cv=min(5, len(valid_classes)), scoring='accuracy')
        cv_acc = scores.mean()

    clf.fit(X_f, y_f)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f'phase_{exercise_id}.joblib'
    joblib.dump({
        'model':        clf,
        'classes':      list(clf.classes_),
        'joints':       JOINTS,
        'exercise_id':  exercise_id,
        'cv_accuracy':  round(float(cv_acc), 3) if cv_acc is not None else None,
        'n_samples':    len(X_f),
        'class_counts': dict(counts),
    }, model_path)

    if verbose:
        acc_str = f'{cv_acc:.3f}' if cv_acc is not None else 'n/a'
        print(f'  [{exercise_id}] {len(X_f)} frames  cv={acc_str}  '
              f'classes={list(clf.classes_)}  → {model_path.name}')
    return model_path


# ---------------------------------------------------------------------------
# Train all (or one) exercises
# ---------------------------------------------------------------------------

def train_all(
    labels_root: Path = LABELS_DIR,
    exercise_filter: Optional[str] = None,
    skip: int = 1,
    verbose: bool = True,
) -> None:
    from core.db import get_db
    ex_model = ExerciseModel.from_mongo(get_db())
    exercise_ids = [exercise_filter] if exercise_filter else ex_model.list_exercises()

    trained = []
    for ex_id in exercise_ids:
        if verbose:
            print(f'\n[{ex_id}]')

        X, y, eval_file, phase_angles = collect_samples_from_labels(ex_id, labels_root, skip, verbose)

        if phase_angles:
            write_exercise_angles(ex_id, phase_angles, verbose)

        path = train_exercise(ex_id, X, y, verbose)

        if path and eval_file is not None:
            if verbose:
                print(f'  Evaluating on held-out: {eval_file.name}')
            from core.ml.eval import evaluate
            evaluate(eval_file, verbose=verbose)

        if path:
            trained.append(ex_id)

    if verbose:
        print(f'\nDone. Trained: {trained or "none"}')
        print(f'Models: {MODELS_DIR}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train per-exercise phase classifiers from hand-labeled videos.'
    )
    parser.add_argument('--labels',   type=Path, default=LABELS_DIR,
                        help=f'Root directory of label JSON files (default: {LABELS_DIR})')
    parser.add_argument('--exercise', type=str,  default=None,
                        help='Train only this exercise ID')
    parser.add_argument('--skip',     type=int,  default=1,
                        help='Frame skip override (default: use value from label file)')
    parser.add_argument('--quiet',    action='store_true')
    args = parser.parse_args()

    if not args.labels.exists():
        print(f'Error: labels directory not found: {args.labels}', file=sys.stderr)
        sys.exit(1)

    train_all(args.labels, args.exercise, args.skip, verbose=not args.quiet)


if __name__ == '__main__':
    main()
