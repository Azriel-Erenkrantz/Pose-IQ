"""
Segment an angle trajectory into movement phases.

Algorithm:
  1. Pick the joint with the highest variance — it moves the most → primary joint.
  2. Smooth its trajectory (moving average).
  3. Find peaks (standing/extended) and valleys (bottom/flexed).
  4. Every segment between consecutive extrema = one phase.
  5. Group phases into reps: peak → valley → peak = one full rep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Phase:
    index:       int
    start_frame: int
    end_frame:   int
    angle_data:  Dict[str, List[float]] = field(default_factory=dict)

    @property
    def stats(self) -> Dict[str, Dict[str, float]]:
        """Per-joint {min, max, mean, std} for this phase."""
        out = {}
        for joint, vals in self.angle_data.items():
            if vals:
                arr = np.array(vals)
                out[joint] = {
                    'min':  round(float(arr.min()), 1),
                    'max':  round(float(arr.max()), 1),
                    'mean': round(float(arr.mean()), 1),
                    'std':  round(float(arr.std()), 1),
                }
        return out


def _smooth(values: List[float], window: int = 7) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def _primary_joint(trajectories: Dict[str, List[float]]) -> str:
    return max(trajectories, key=lambda k: float(np.var(trajectories[k])))


def _find_extrema(signal: np.ndarray, min_gap: int = 8) -> Tuple[List[int], List[int]]:
    """Return (peaks, valleys) indices, enforcing a minimum gap between them."""
    peaks: List[int]   = []
    valleys: List[int] = []

    for i in range(1, len(signal) - 1):
        is_peak   = signal[i] > signal[i - 1] and signal[i] > signal[i + 1]
        is_valley = signal[i] < signal[i - 1] and signal[i] < signal[i + 1]

        if is_peak and (not peaks or i - peaks[-1] >= min_gap):
            peaks.append(i)
        elif is_valley and (not valleys or i - valleys[-1] >= min_gap):
            valleys.append(i)

    return peaks, valleys


def _slice(trajectories: Dict[str, List[float]], start: int, end: int) -> Dict[str, List[float]]:
    return {k: v[start:end + 1] for k, v in trajectories.items()}


def segment_reps(
    trajectories: Dict[str, List[float]],
    min_phase_frames: int = 10,
) -> List[List[Phase]]:
    """
    Segment angle trajectories into reps, each rep a list of Phases.

    Returns:
        List of reps. Each rep = [Phase(descent), Phase(bottom?), Phase(ascent)].
        If only one rep is detected (short video), returns one rep.
    """
    if not trajectories:
        return []

    primary = _primary_joint(trajectories)
    signal  = _smooth(trajectories[primary])
    peaks, valleys = _find_extrema(signal)

    # Merge and sort all boundary frames
    boundaries = sorted(set(peaks + valleys))

    if len(boundaries) < 2:
        # Can't segment — treat whole video as a single phase
        whole = Phase(0, 0, len(signal) - 1, _slice(trajectories, 0, len(signal) - 1))
        return [[whole]]

    # Build phases from consecutive boundaries
    all_phases: List[Phase] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < min_phase_frames:
            continue
        all_phases.append(Phase(
            index       = len(all_phases),
            start_frame = s,
            end_frame   = e,
            angle_data  = _slice(trajectories, s, e),
        ))

    if not all_phases:
        return []

    # Group into reps: a rep ends when we return to a peak (back to standing)
    reps: List[List[Phase]] = []
    current_rep: List[Phase] = []

    for phase in all_phases:
        current_rep.append(phase)
        if phase.end_frame in peaks and len(current_rep) >= 2:
            reps.append(current_rep)
            current_rep = []

    if current_rep:
        reps.append(current_rep)

    return reps
