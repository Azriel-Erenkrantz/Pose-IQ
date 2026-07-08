"""
Model 2 builder: reference videos → MongoDB exercise_angles collection.

Reads exercise phase definitions (motion_direction, is_initial, diagnostic_joints)
from MongoDB exercises collection, processes reference videos to measure angle
ranges per phase, and writes min/max/mean/std to MongoDB exercise_angles collection.

Phase assignment uses motion_direction matching:
  is_initial + stable  → frames near peaks (standing/start position)
  decreasing           → frames from peak to valley (descent)
  non-initial + stable → frames near valleys (hold at bottom)
  increasing           → frames from valley to peak (ascent)

Usage:
    python -m core.model2.builder data/videos/
    python -m core.model2.builder data/videos/ --exercise squat --skip 2
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .angles import angles_over_time
from .extractor import extract_frames
from .phases import _find_extrema, _smooth

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
STABLE_WINDOW = 5    # frames around a peak/valley assigned to stable phases
MIN_SEGMENT_FRAMES = 10  # ignore segments shorter than this


def _exercise_id(directory: Path) -> str:
    return directory.name.lower().replace(' ', '_').replace('-', '_')


def _primary_joint(ex_doc: dict, trajectories: Dict[str, List[float]]) -> Optional[str]:
    """
    Pick the diagnostic joint with the highest variance in this video.
    Falls back to the overall highest-variance joint if none are visible.
    """
    candidates: set[str] = set()
    for phase in ex_doc.get('phases', []):
        for j in phase.get('diagnostic_joints', []):
            candidates.add(j)

    available = {j: trajectories[j] for j in candidates if j in trajectories}
    if available:
        return max(available, key=lambda k: float(np.var(available[k])))

    if not trajectories:
        return None
    return max(trajectories, key=lambda k: float(np.var(trajectories[k])))


def _segment_and_label(
    trajectories: Dict[str, List[float]],
    phase_defs: List[dict],
    primary: str,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Assign trajectory frames to named phases by motion_direction.
    Returns {phase_name: {joint: [angle_values]}}.
    """
    if primary not in trajectories:
        return {}

    signal = _smooth(trajectories[primary])
    peaks, valleys = _find_extrema(signal)

    if not peaks and not valleys:
        return {}

    extrema = sorted([(i, 'peak') for i in peaks] + [(i, 'valley') for i in valleys])

    stable_init   = [p for p in phase_defs if p.get('is_initial') and p.get('motion_direction') == 'stable']
    stable_hold   = [p for p in phase_defs if not p.get('is_initial') and p.get('motion_direction') == 'stable']
    decreasing    = [p for p in phase_defs if p.get('motion_direction') == 'decreasing']
    increasing    = [p for p in phase_defs if p.get('motion_direction') == 'increasing']

    result: Dict[str, Dict[str, List[float]]] = {
        p['name']: {j: [] for j in trajectories} for p in phase_defs
    }

    n = len(signal)

    # Directional segments between consecutive extrema
    for i in range(len(extrema) - 1):
        idx_s, kind_s = extrema[i]
        idx_e, kind_e = extrema[i + 1]

        if idx_e - idx_s < MIN_SEGMENT_FRAMES:
            continue

        if kind_s == 'peak' and kind_e == 'valley':
            target = decreasing
        elif kind_s == 'valley' and kind_e == 'peak':
            target = increasing
        else:
            continue

        for p in target:
            for joint, vals in trajectories.items():
                result[p['name']][joint].extend(vals[idx_s:idx_e + 1])

    # Stable windows around each extremum
    for idx, kind in extrema:
        s = max(0, idx - STABLE_WINDOW)
        e = min(n - 1, idx + STABLE_WINDOW)

        target = stable_init if kind == 'peak' else stable_hold
        for p in target:
            for joint, vals in trajectories.items():
                result[p['name']][joint].extend(vals[s:e + 1])

    return result


def _compute_stats(phase_data: Dict[str, List[float]]) -> Dict[str, dict]:
    """Compute min/max/mean/std per joint from collected frames."""
    stats = {}
    for joint, vals in phase_data.items():
        if len(vals) < 3:
            continue
        arr = np.array(vals, dtype=float)
        stats[joint] = {
            'min':      round(float(arr.min()), 1),
            'max':      round(float(arr.max()), 1),
            'mean':     round(float(arr.mean()), 1),
            'std':      round(float(arr.std()), 1),
            'n_frames': len(vals),
        }
    return stats


def process_exercise(
    exercise_dir: Path,
    ex_doc: dict,
    skip: int = 1,
    verbose: bool = False,
) -> Dict[str, dict]:
    """
    Process all videos for one exercise.
    Returns {phase_name: {joint: {min, max, mean, std, n_frames}}}.
    """
    video_files = sorted(f for f in exercise_dir.rglob('*') if f.suffix.lower() in VIDEO_SUFFIXES)
    if not video_files:
        if verbose:
            print('  no video files')
        return {}

    phase_defs = ex_doc.get('phases', [])
    if not phase_defs:
        if verbose:
            print('  no phases defined')
        return {}

    # Accumulate frames per phase across all videos
    accumulated: Dict[str, Dict[str, List[float]]] = {
        p['name']: {} for p in phase_defs
    }

    for vf in video_files:
        if verbose:
            print(f'  {vf.name} ...', end=' ', flush=True)
        try:
            frames = extract_frames(vf, skip=skip)
            if len(frames) < 20:
                if verbose:
                    print(f'too short ({len(frames)} frames), skipped')
                continue

            trajectories = angles_over_time(frames)
            if not trajectories:
                if verbose:
                    print('no visible joints, skipped')
                continue

            primary = _primary_joint(ex_doc, trajectories)
            if primary is None:
                if verbose:
                    print('no primary joint, skipped')
                continue

            labeled = _segment_and_label(trajectories, phase_defs, primary)

            populated = [k for k, v in labeled.items() if any(len(vv) > 0 for vv in v.values())]
            if verbose:
                print(f'{len(frames)} frames → phases: {populated}')

            for phase_name, joint_data in labeled.items():
                for joint, vals in joint_data.items():
                    if vals:
                        if joint not in accumulated[phase_name]:
                            accumulated[phase_name][joint] = []
                        accumulated[phase_name][joint].extend(vals)

        except Exception as exc:
            if verbose:
                print(f'ERROR: {exc}')

    result = {}
    for phase_name, joint_data in accumulated.items():
        stats = _compute_stats(joint_data)
        if stats:
            result[phase_name] = stats

    return result


def build(
    videos_root: Path,
    exercise_filter: Optional[str] = None,
    skip: int = 1,
    verbose: bool = True,
) -> None:
    from core.db import get_db
    db = get_db()

    query = {'id': exercise_filter} if exercise_filter else {}
    ex_docs = list(db.exercises.find(query))

    if not ex_docs:
        msg = (f'exercise "{exercise_filter}" not found in MongoDB'
               if exercise_filter
               else 'no exercises in MongoDB — run: python -m core.exercise.seed')
        print(f'Error: {msg}', file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc)
    updated: list[str] = []

    for ex_doc in ex_docs:
        ex_id = ex_doc['id']
        if verbose:
            print(f'\n[{ex_id}]')

        ex_dir = videos_root / ex_id
        if not ex_dir.exists():
            candidates = [
                d for d in videos_root.iterdir()
                if d.is_dir() and d.name.lower().replace(' ', '_').replace('-', '_') == ex_id
            ]
            ex_dir = candidates[0] if candidates else None

        if ex_dir is None:
            if verbose:
                print('  no video directory found')
            continue

        phase_stats = process_exercise(ex_dir, ex_doc, skip=skip, verbose=verbose)

        if not phase_stats:
            if verbose:
                print('  no angle data extracted')
            continue

        for phase_name, joint_stats in phase_stats.items():
            db.exercise_angles.replace_one(
                {'exercise_id': ex_id, 'phase': phase_name},
                {
                    'exercise_id': ex_id,
                    'phase':       phase_name,
                    'joints':      joint_stats,
                    'updated_at':  now,
                },
                upsert=True,
            )

        updated.append(ex_id)
        if verbose:
            for phase_name, joint_stats in phase_stats.items():
                summary = ', '.join(
                    f'{j}={v["mean"]:.0f}°±{v["std"]:.0f}°'
                    for j, v in joint_stats.items()
                )
                print(f'  [{phase_name}] {summary}')

    if verbose:
        print(f'\nUpdated exercise_angles for: {updated or "none"}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure angle ranges from reference videos → MongoDB exercise_angles'
    )
    parser.add_argument('videos_dir', type=Path)
    parser.add_argument('--exercise', type=str, default=None,
                        help='Process only this exercise ID')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process every Nth frame (default 1). Use 2-3 for speed.')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if not args.videos_dir.exists():
        print(f'Error: {args.videos_dir} does not exist.', file=sys.stderr)
        sys.exit(1)

    build(args.videos_dir, args.exercise, skip=args.skip, verbose=not args.quiet)


if __name__ == '__main__':
    main()
