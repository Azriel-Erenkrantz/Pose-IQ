"""
Evaluate the trained RF phase classifier against a hand-labeled test video.

Label-file format (JSON):
----------------------------------------------------------------------
{
  "exercise_id": "squat",
  "video": "data/test_videos/squat_good.mp4",
  "skip": 2,
  "labels": [
    {"start_sec": 0.0,  "end_sec": 1.5,  "phase": "standing"},
    {"start_sec": 1.5,  "end_sec": 3.8,  "phase": "descending"},
    {"start_sec": 3.8,  "end_sec": 4.9,  "phase": "hold"},
    {"start_sec": 4.9,  "end_sec": 7.2,  "phase": "ascending"},
    {"start_sec": 7.2,  "end_sec": 8.5,  "phase": "standing"}
  ]
}
----------------------------------------------------------------------

"start_sec" / "end_sec" are the timestamps shown in any video player (VLC,
QuickTime, etc.). You can also use "start_frame" / "end_frame" for raw video
frame numbers if you prefer counting frames directly.

Usage:
    python -m core.ml.eval data/labeled_vidz/squat_test.json
    python -m core.ml.eval data/labeled_vidz/squat_test.json --gaussian
    python -m core.ml.eval data/labeled_vidz/squat_test.json --skip 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DELTA_WINDOW = 5  # rolling average window for angle delta feature


def _video_fps(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def _resolve_video(video_str: str, label_file: Path) -> Optional[Path]:
    """Resolve video path: absolute → relative to label dir → relative to project root."""
    p = Path(video_str)
    if p.is_absolute() and p.exists():
        return p
    candidate = label_file.parent / p
    if candidate.exists():
        return candidate
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    return None


def _build_frame_labels(
    labels: List[dict],
    n_frames: int,
    fps: float,
    skip: int,
) -> Dict[int, str]:
    """
    Map extracted frame indices → phase names.

    Labels can specify time ("start_sec"/"end_sec") or raw video frame numbers
    ("start_frame"/"end_frame"). Time is preferred since any video player shows it.

    Extracted frame index = floor(raw_video_frame / skip)
                          = floor(timestamp_sec * fps / skip)
    """
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


def evaluate(
    label_path: Path,
    skip_override: Optional[int] = None,
    use_gaussian: bool = False,
    verbose: bool = True,
    rep_tol_sec: float = 0.5,
) -> dict:
    data = json.loads(label_path.read_text(encoding='utf-8'))

    exercise_id = data['exercise_id']
    skip = skip_override if skip_override is not None else data.get('skip', 1)
    raw_labels = data['labels']

    video_path = _resolve_video(data['video'], label_path)
    if video_path is None:
        print(f'Error: video not found: {data["video"]}', file=sys.stderr)
        print(f'  Tried: {label_path.parent / data["video"]}', file=sys.stderr)
        print(f'  Tried: {PROJECT_ROOT / data["video"]}', file=sys.stderr)
        sys.exit(1)

    from core.db import get_db
    from core.exercise.exercise_model import ExerciseModel
    from core.ml.angles import angles_over_time, compute_angles
    from core.ml.extractor import extract_frames
    from core.ml.classifier import classify, classify_trained
    from core.ml.reps import rep_metrics
    from core.ml.smoother import PhaseSmoother
    from core.ml.trainer import rolling_deltas

    db = get_db()
    model = ExerciseModel.from_mongo(db)
    exercise = model.get_exercise(exercise_id)
    if exercise is None:
        print(f'Error: exercise "{exercise_id}" not found in MongoDB', file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f'\nEvaluating: {exercise.name}')
        print(f'Video  : {video_path.name}')
        print(f'Skip   : {skip}')
        print(f'Method : {"gaussian" if use_gaussian else "rf (fallback: gaussian)"}')

    frames = extract_frames(video_path, skip=skip)
    if len(frames) < 5:
        print(f'Error: only {len(frames)} frames extracted — is the video readable?', file=sys.stderr)
        sys.exit(1)

    per_frame: List[Dict[str, float]] = [compute_angles(f) for f in frames]
    trajectories = angles_over_time(frames)

    # Primary joint: highest variance among exercise's primary diagnostic joints
    primary: Optional[str] = None
    if exercise.primary_joints:
        candidates = {j: trajectories[j] for j in exercise.primary_joints if j in trajectories}
        if candidates:
            primary = max(candidates, key=lambda k: float(np.var(candidates[k])))

    fps = _video_fps(video_path)

    if verbose:
        print(f'Frames : {len(frames)}  (fps={fps:.1f}, primary_joint={primary})')

    frame_labels = _build_frame_labels(raw_labels, len(frames), fps, skip)
    if not frame_labels:
        print('Error: no labeled frames found — check start_sec/end_sec values', file=sys.stderr)
        sys.exit(1)

    all_phases = sorted({p.name for p in exercise.phases})
    method_used = 'gaussian'

    # ── Pass 1: predict EVERY frame (the smoother needs temporal continuity),
    #    then score only the labeled ones.
    raw_preds:      List[str] = []
    smoothed_preds: List[str] = []
    smoother = PhaseSmoother(exercise)
    rate = fps / skip   # extracted frames per second → deltas in °/sec

    for i, angles in enumerate(per_frame):
        deltas = rolling_deltas(per_frame, i, rate=rate)
        prev_angle = per_frame[i - 1].get(primary) if i > 0 and primary else None
        curr_angle = angles.get(primary) if primary else None

        if use_gaussian:
            predicted_phase = classify(exercise, angles, prev_angle, curr_angle)
            method_used = 'gaussian'
        else:
            predicted_phase, method_used = classify_trained(
                exercise, angles, prev_angle, curr_angle, deltas=deltas
            )
        name = predicted_phase.name if predicted_phase else 'unknown'
        raw_preds.append(name)
        smoothed_preds.append(smoother.update(name) or 'unknown')

    # ── Pass 2: score. Boundary-tolerant = prediction matches the label of ANY
    #    frame within ±0.3s — phase boundaries are ambiguous even for humans.
    tol = max(1, int(round(0.3 * fps / skip)))

    def _tolerant_hit(idx: int, predicted: str) -> bool:
        return any(
            frame_labels.get(k) == predicted
            for k in range(idx - tol, idx + tol + 1)
        )

    correct:   Dict[str, int] = defaultdict(int)   # raw
    correct_s: Dict[str, int] = defaultdict(int)   # smoothed
    total:     Dict[str, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = {p: defaultdict(int) for p in all_phases}
    raw_tol = smooth_tol = 0

    for i in sorted(frame_labels):
        if i >= len(raw_preds):
            break
        truth = frame_labels[i]
        raw, smooth = raw_preds[i], smoothed_preds[i]

        total[truth] += 1
        if truth in confusion:
            confusion[truth][raw] += 1
        if raw == truth:
            correct[truth] += 1
        if smooth == truth:
            correct_s[truth] += 1
        if _tolerant_hit(i, raw):
            raw_tol += 1
        if _tolerant_hit(i, smooth):
            smooth_tol += 1

    # ── Pass 3: rep-level. This is the product metric — the app counts reps
    #    and coaches per rep, so "was each rep detected once, at roughly the
    #    right time" matters more than frame-exact boundaries.
    cycle = [p.name for p in sorted(exercise.phases, key=lambda p: p.order)]
    truth_seq = [frame_labels.get(i) for i in range(len(frames))]
    rep_tol = max(1, int(round(rep_tol_sec * fps / skip)))
    reps_raw      = rep_metrics(truth_seq, raw_preds, cycle, rep_tol)
    reps_smoothed = rep_metrics(truth_seq, smoothed_preds, cycle, rep_tol)

    overall_total     = sum(total.values())
    overall_correct   = sum(correct.values())
    overall_correct_s = sum(correct_s.values())
    overall_acc        = overall_correct / overall_total if overall_total else 0.0
    overall_acc_s      = overall_correct_s / overall_total if overall_total else 0.0
    overall_acc_tol    = raw_tol / overall_total if overall_total else 0.0
    overall_acc_s_tol  = smooth_tol / overall_total if overall_total else 0.0

    if verbose:
        print(f'\n{"─" * 54}')
        print(f'Method : {method_used}   (boundary tolerance: ±0.3s = ±{tol} frames)')
        print(f'Overall raw       : {overall_correct}/{overall_total}  ({100*overall_acc:.1f}%)')
        print(f'Overall smoothed  : {overall_correct_s}/{overall_total}  ({100*overall_acc_s:.1f}%)')
        print(f'Raw      ±0.3s    : {raw_tol}/{overall_total}  ({100*overall_acc_tol:.1f}%)')
        print(f'Smoothed ±0.3s    : {smooth_tol}/{overall_total}  ({100*overall_acc_s_tol:.1f}%)')
        print(f'\nPer-phase accuracy (raw → smoothed):')

        labeled_phases = sorted(total.keys())
        for phase in labeled_phases:
            n = total[phase]
            c  = correct.get(phase, 0)
            cs = correct_s.get(phase, 0)
            filled = int(20 * cs / n)
            bar = '█' * filled + '░' * (20 - filled)
            print(f'  {phase:20s}  {100*c/n:5.1f}% → {100*cs/n:5.1f}%  {bar}')

        print(f'\nRep-level (anchor tolerance: ±{rep_tol_sec:.1f}s = ±{rep_tol} frames):')
        print(f'  {"":10s}{"truth":>7s}{"pred":>7s}{"match":>7s}{"miss":>7s}{"extra":>7s}{"recall":>9s}{"F1":>7s}')
        for label, m in (('raw', reps_raw), ('smoothed', reps_smoothed)):
            print(f'  {label:10s}{m["truth_reps"]:>7d}{m["predicted_reps"]:>7d}'
                  f'{m["matched"]:>7d}{m["missed"]:>7d}{m["extra"]:>7d}'
                  f'{100*m["recall"]:>8.1f}%{m["f1"]:>7.2f}')

        print(f'\nConfusion matrix (row=truth, col=predicted):')
        col_w = max(14, max(len(p) for p in all_phases) + 2)
        header = f'{"":20s}' + ''.join(f'{p[:col_w]:>{col_w}}' for p in all_phases)
        print(header)
        for truth in labeled_phases:
            row_vals = ''.join(
                f'{confusion[truth].get(pred, 0):>{col_w}d}' for pred in all_phases
            )
            print(f'{truth:20s}{row_vals}')
        print(f'{"─" * 54}')

    return {
        'exercise_id':    exercise_id,
        'video':          str(video_path),
        'method':         method_used,
        'accuracy':       round(overall_acc, 3),
        'accuracy_smoothed':          round(overall_acc_s, 3),
        'accuracy_tolerant':          round(overall_acc_tol, 3),
        'accuracy_smoothed_tolerant': round(overall_acc_s_tol, 3),
        'tolerance_frames': tol,
        'labeled_frames': overall_total,
        'per_phase': {
            p: round(correct.get(p, 0) / total[p], 3)
            for p in sorted(total.keys())
        },
        'per_phase_smoothed': {
            p: round(correct_s.get(p, 0) / total[p], 3)
            for p in sorted(total.keys())
        },
        'confusion': {t: dict(confusion[t]) for t in sorted(total.keys())},
        'reps': {
            'tolerance_sec': rep_tol_sec,
            'raw': reps_raw,
            'smoothed': reps_smoothed,
        },
    }


def main() -> None:
    # Windows consoles may default to cp1252, which can't print the project's
    # non-ASCII path — degrade unprintable characters instead of crashing.
    sys.stdout.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(
        description='Evaluate the trained RF phase classifier against a labeled test video.'
    )
    parser.add_argument('label_file', type=Path,
                        help='Path to the label JSON file')
    parser.add_argument('--skip', type=int, default=None,
                        help='Override the skip value in the label file')
    parser.add_argument('--gaussian', action='store_true',
                        help='Use Gaussian classifier instead of trained RF')
    parser.add_argument('--rep-tol', type=float, default=0.5,
                        help='Rep anchor matching tolerance in seconds (default 0.5)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if not args.label_file.exists():
        print(f'Error: label file not found: {args.label_file}', file=sys.stderr)
        sys.exit(1)

    evaluate(
        args.label_file,
        skip_override=args.skip,
        use_gaussian=args.gaussian,
        verbose=not args.quiet,
        rep_tol_sec=args.rep_tol,
    )


if __name__ == '__main__':
    main()
