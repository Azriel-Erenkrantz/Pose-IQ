"""
Phase classifier for Model 1.

Given the current joint angles and an Exercise definition, scores every phase
and returns the most likely one.

Scoring strategy per phase:
  - For each diagnostic joint, compute a likelihood score (0-1).
  - If the joint has a measured mean/std from reference videos → Gaussian score.
  - Otherwise → binary: 1.0 if inside [min, max], 0.0 if outside.
  - Missing joints contribute 0.0 to the score (treated as uncertainty, not error).
  - Final phase score = average across all diagnostic joints.
  - The phase with the highest score wins.

motion_direction is used as a tiebreaker when two phases have identical angle
ranges (e.g. descending vs ascending in a squat both use 90-155°). The caller
must supply the previous primary-joint angle so the direction can be inferred.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.exercise.exercise_model import Exercise, Phase

MODELS_DIR = Path(__file__).parent.parent.parent / 'data' / 'models'


@dataclass
class PhaseScore:
    phase: Phase
    score: float          # 0.0 – 1.0
    matched_joints: int   # how many diagnostic joints contributed
    total_joints: int     # total diagnostic joints for this phase


def _gaussian_score(value: float, mean: float, std: float) -> float:
    """Probability density of value under N(mean, std), normalised to peak at 1.0."""
    if std <= 0:
        return 1.0 if abs(value - mean) < 1e-6 else 0.0
    return math.exp(-0.5 * ((value - mean) / std) ** 2)


def _boundary_score(value: float, lo: float, hi: float) -> float:
    """1.0 inside [lo, hi], decays linearly outside (reaches 0 at 2× the range away)."""
    if lo <= value <= hi:
        return 1.0
    span = max(hi - lo, 1.0)
    dist = min(abs(value - lo), abs(value - hi))
    return max(0.0, 1.0 - dist / span)


def score_phase(phase: Phase, angles: Dict[str, float]) -> PhaseScore:
    """Score a single phase against current joint angles."""
    joints_to_check = phase.diagnostic_joints or list(phase.angles.keys())
    total = len(joints_to_check)
    if total == 0:
        return PhaseScore(phase, 0.0, 0, 0)

    scores: List[float] = []
    matched = 0

    for joint in joints_to_check:
        ar = phase.angles.get(joint)
        value = angles.get(joint)

        if ar is None or value is None:
            scores.append(0.0)
            continue

        matched += 1
        if ar.mean is not None and ar.std is not None and ar.std > 0:
            scores.append(_gaussian_score(value, ar.mean, ar.std))
        else:
            scores.append(_boundary_score(value, ar.min, ar.max))

    avg = sum(scores) / total if scores else 0.0
    return PhaseScore(phase, avg, matched, total)


def classify(
    exercise: Exercise,
    angles: Dict[str, float],
    prev_primary_angle: Optional[float] = None,
    current_primary_angle: Optional[float] = None,
    observed_delta: Optional[float] = None,
) -> Optional[Phase]:
    """
    Return the most likely Phase for the given joint angles.

    Motion direction (for tie-breaking phases that share the same angle range,
    e.g. descending vs ascending) can come from either:
      - observed_delta: a pre-computed primary-joint delta, or
      - prev_primary_angle / current_primary_angle: consecutive readings.
    """
    if not exercise.phases:
        return None

    scored = [score_phase(p, angles) for p in exercise.phases]
    scored.sort(key=lambda s: s.score, reverse=True)

    best = scored[0]

    # Tiebreak: if top two scores are within 5% of each other and their
    # motion_direction differs, use the observed direction to pick.
    if len(scored) > 1 and (best.score - scored[1].score) < 0.05:
        runner_up = scored[1]
        if observed_delta is None and (
            prev_primary_angle is not None and current_primary_angle is not None
        ):
            observed_delta = current_primary_angle - prev_primary_angle
        if (
            best.phase.motion_direction != runner_up.phase.motion_direction
            and observed_delta is not None
        ):
            if observed_delta < -1.0:
                observed_dir = 'decreasing'
            elif observed_delta > 1.0:
                observed_dir = 'increasing'
            else:
                observed_dir = 'stable'

            if runner_up.phase.motion_direction == observed_dir:
                return runner_up.phase

    return best.phase


def all_scores(exercise: Exercise, angles: Dict[str, float]) -> List[PhaseScore]:
    """Return scores for every phase, sorted best first. Useful for debugging."""
    scored = [score_phase(p, angles) for p in exercise.phases]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Trained model loader (used by classify_trained below)
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Optional[dict]] = {}


def _load_model(exercise_id: str) -> Optional[dict]:
    if exercise_id not in _model_cache:
        path = MODELS_DIR / f'phase_{exercise_id}.joblib'
        if path.exists():
            try:
                import joblib
                _model_cache[exercise_id] = joblib.load(path)
            except Exception:
                _model_cache[exercise_id] = None
        else:
            _model_cache[exercise_id] = None
    return _model_cache[exercise_id]


def classify_trained(
    exercise: Exercise,
    angles: Dict[str, float],
    prev_primary_angle: Optional[float] = None,
    current_primary_angle: Optional[float] = None,
    rolling_delta: Optional[float] = None,
    deltas: Optional[Dict[str, float]] = None,
) -> tuple[Optional[Phase], str]:
    """
    Classify using the trained Random Forest when available.
    Falls back to Gaussian/boundary scoring if no model exists yet.

    deltas: per-joint rolling deltas (trainer.rolling_deltas / DeltaTracker) —
    required for full accuracy with feature_version 2 models.
    rolling_delta / prev+current primary angle: legacy single-delta signals,
    used by feature_version 1 models and the Gaussian direction tiebreak.

    Returns (Phase, method) where method is 'rf' or 'gaussian'.
    """
    bundle = _load_model(exercise.id)
    if bundle is not None:
        from core.ml.trainer import JOINTS, angles_to_features

        if bundle.get('feature_version', 1) >= 2:
            features = angles_to_features(angles, deltas).reshape(1, -1)
        else:
            # Legacy 13-feature bundle: 12 angles + one primary-joint delta.
            if rolling_delta is not None:
                delta = rolling_delta
            elif prev_primary_angle is not None and current_primary_angle is not None:
                delta = current_primary_angle - prev_primary_angle
            else:
                delta = 0.0
            features = np.array(
                [[angles.get(j, -1.0) for j in JOINTS] + [delta]], dtype=np.float32
            )
        phase_name: str = bundle['model'].predict(features)[0]
        phase = exercise.get_phase(phase_name)
        if phase is not None:
            return phase, 'rf'

    # Fallback to Gaussian classifier. Direction for the tiebreak: explicit
    # rolling_delta, else the primary joint's entry in the deltas dict.
    observed = rolling_delta
    if observed is None and deltas:
        for j in exercise.primary_joints:
            if j in deltas:
                observed = deltas[j]
                break
    return classify(exercise, angles, prev_primary_angle, current_primary_angle,
                    observed_delta=observed), 'gaussian'


class DeltaTracker:
    """
    Per-joint rolling angular velocities for a live frame stream.

    Produces the same numbers as trainer.rolling_deltas() computes offline
    (degrees/second), so live inference sees the exact feature distribution
    the model trained on — regardless of the camera's actual frame rate.
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._prev: Optional[Dict[str, float]] = None
        self._prev_t: Optional[float] = None
        self._history: Dict[str, deque] = {}

    def update(self, angles: Dict[str, float], now: Optional[float] = None) -> Dict[str, float]:
        """
        Feed the current frame's angles; returns {joint: mean velocity}.
        now: wall-clock seconds (time.time()) — required for °/sec output;
        without it, deltas are per-frame (only valid for fixed-rate sources).
        """
        if self._prev is not None:
            dt = 1.0
            if now is not None and self._prev_t is not None:
                dt = now - self._prev_t
            if dt > 0:
                for joint, value in angles.items():
                    if joint in self._prev:
                        self._history.setdefault(
                            joint, deque(maxlen=self._window)
                        ).append((value - self._prev[joint]) / dt)
        self._prev = dict(angles)
        self._prev_t = now
        return {j: sum(d) / len(d) for j, d in self._history.items() if d}
