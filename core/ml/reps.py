"""
Rep-level metrics for phase-sequence predictions.

Frame-exact accuracy punishes boundary disagreements that don't matter to the
product: the app counts reps and coaches per rep, and hand labels themselves
are only accurate to a few frames around each transition. This module scores
what the product actually does — did each rep get detected, once, at roughly
the right time?

Definitions
-----------
The exercise cycle is its phases sorted by `order`:
    cycle[0]  = resting/initial phase (e.g. standing)
    cycle[1]  = first movement phase  (e.g. descending)
    cycle[-1] = final/return phase    (e.g. ascending)

A **rep** completes at the transition INTO the final phase, provided the
first movement phase occurred since the previous rep. That transition frame
is the rep's **anchor** — a single well-defined event per rep (the turnaround
point), robust to fuzzy boundaries and skipped optional phases (e.g. a squat
with no visible 'hold').

Truth and predicted anchors are matched greedily by smallest time offset
within a tolerance window. From the matching we report:
    matched  — reps the model detected     (→ recall = matched / truth reps)
    missed   — truth reps with no match
    extra    — predicted reps with no match (double counts / phantom reps)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

UNKNOWN = ('unknown', None)


@dataclass
class Rep:
    start_idx: int    # frame where the movement (cycle[1]) run began
    anchor_idx: int   # frame of the transition into cycle[-1]
    end_idx: int      # last frame of the cycle[-1] segment


def _segments(phases: Sequence[Optional[str]]) -> List[Tuple[str, int, int]]:
    """Compress a per-frame phase sequence into (phase, start_idx, end_idx)
    segments. Frames labeled 'unknown'/None carry no information and are
    skipped — a segment interrupted only by unknowns stays one segment."""
    segs: List[Tuple[str, int, int]] = []
    for i, p in enumerate(phases):
        if p in UNKNOWN:
            continue
        if segs and segs[-1][0] == p:
            segs[-1] = (p, segs[-1][1], i)
        else:
            segs.append((p, i, i))
    return segs


def extract_reps(phases: Sequence[Optional[str]], cycle: Sequence[str]) -> List[Rep]:
    """Extract completed reps from a per-frame phase sequence.

    A rep is anchored at each entry into cycle[-1] that was preceded by
    cycle[1] since the last anchor. An isolated final-phase blip with no
    movement phase before it (labeling artifact or model flicker) is not
    a rep.
    """
    if len(cycle) < 2:
        return []
    first_move, final = cycle[1], cycle[-1]

    reps: List[Rep] = []
    pending_start: Optional[int] = None
    for phase, start, end in _segments(phases):
        if phase == first_move:
            pending_start = start
        if phase == final and pending_start is not None:
            reps.append(Rep(start_idx=pending_start, anchor_idx=start, end_idx=end))
            pending_start = None
    return reps


def match_reps(
    truth_anchors: Sequence[int],
    pred_anchors: Sequence[int],
    tol_frames: int,
) -> List[Tuple[int, int]]:
    """Greedily pair truth and predicted anchors by smallest |offset| within
    the tolerance. Each anchor is used at most once. Returns matched
    (truth_idx, pred_idx) pairs into the input sequences."""
    candidates = sorted(
        ((abs(t - p), ti, pi)
         for ti, t in enumerate(truth_anchors)
         for pi, p in enumerate(pred_anchors)
         if abs(t - p) <= tol_frames),
        key=lambda c: c[0],
    )
    used_t: set = set()
    used_p: set = set()
    pairs: List[Tuple[int, int]] = []
    for _, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        pairs.append((ti, pi))
    return pairs


def rep_metrics(
    truth_phases: Sequence[Optional[str]],
    pred_phases: Sequence[Optional[str]],
    cycle: Sequence[str],
    tol_frames: int,
) -> Dict:
    """Compare predicted reps against truth reps; both sequences are indexed
    in the same (extracted) frame space."""
    truth_reps = extract_reps(truth_phases, cycle)
    pred_reps = extract_reps(pred_phases, cycle)
    pairs = match_reps(
        [r.anchor_idx for r in truth_reps],
        [r.anchor_idx for r in pred_reps],
        tol_frames,
    )

    matched = len(pairs)
    missed = len(truth_reps) - matched
    extra = len(pred_reps) - matched
    recall = matched / len(truth_reps) if truth_reps else 0.0
    precision = matched / len(pred_reps) if pred_reps else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    offsets = [abs(truth_reps[ti].anchor_idx - pred_reps[pi].anchor_idx)
               for ti, pi in pairs]

    return {
        'truth_reps': len(truth_reps),
        'predicted_reps': len(pred_reps),
        'matched': matched,
        'missed': missed,
        'extra': extra,
        'recall': round(recall, 3),
        'precision': round(precision, 3),
        'f1': round(f1, 3),
        'mean_anchor_offset_frames': (
            round(sum(offsets) / len(offsets), 1) if offsets else None
        ),
        'tolerance_frames': tol_frames,
    }
