"""
Pose landmarks → joint angles (degrees).

Each angle is defined by three joints: A - VERTEX - C.
The returned angle is always at the VERTEX, between 0° and 180°.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .extractor import LANDMARK_NAMES, PoseFrame

# (joint_a, vertex, joint_c) — angle measured at vertex
ANGLE_DEFS: Dict[str, tuple[str, str, str]] = {
    'left_knee':      ('left_hip',       'left_knee',      'left_ankle'),
    'right_knee':     ('right_hip',      'right_knee',     'right_ankle'),
    'left_hip':       ('left_shoulder',  'left_hip',       'left_knee'),
    'right_hip':      ('right_shoulder', 'right_hip',      'right_knee'),
    'left_elbow':     ('left_shoulder',  'left_elbow',     'left_wrist'),
    'right_elbow':    ('right_shoulder', 'right_elbow',    'right_wrist'),
    'left_shoulder':  ('left_hip',       'left_shoulder',  'left_elbow'),
    'right_shoulder': ('right_hip',      'right_shoulder', 'right_elbow'),
    'left_ankle':     ('left_knee',      'left_ankle',     'left_foot_index'),
    'right_ankle':    ('right_knee',     'right_ankle',    'right_foot_index'),
}

MIN_VISIBILITY = 0.4  # ignore joints the model isn't confident about

# Spine needs a stricter bar than the 3-point angles above: it's computed
# from just two real landmarks (shoulder+hip) against a synthetic reference
# point, with no third real landmark to anchor it — more sensitive to a
# landmark's *position* being unreliable even while visibility still clears
# 0.4 (confirmed live 2026-08-27 in the TS port, frontend/src/pose/angles.ts).
MIN_VISIBILITY_SPINE = 0.6

# How far "up" (in normalized image-space y) the synthetic vertical
# reference point sits above the hip. The exact magnitude doesn't affect the
# resulting angle at all — _angle_deg only uses vector *direction*, not
# length — so this just needs to be a small positive number.
SPINE_UP_DELTA = 0.1


def _angle_deg(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at b in the A-B-C triangle, using only x/y (2D)."""
    a2 = np.array(a[:2])
    b2 = np.array(b[:2])
    c2 = np.array(c[:2])
    ba = a2 - b2
    bc = c2 - b2
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-9:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def compute_angles(frame: PoseFrame) -> Dict[str, float]:
    """Return {angle_name: degrees} for all angles visible in this frame."""
    result: Dict[str, float] = {}
    lm  = frame.landmarks
    vis = frame.visibility

    for name, (a, b, c) in ANGLE_DEFS.items():
        if all(vis.get(j, 0.0) >= MIN_VISIBILITY for j in (a, b, c)):
            result[name] = _angle_deg(lm[a], lm[b], lm[c])

    # Spine: angle at the hip between the shoulder and a synthetic
    # straight-up reference point — normalized 2D, like every other angle
    # above (added 2026-08-30; previously computed only in the TS live path,
    # in pixel-space 3D against the original desktop pipeline's fixed-50px
    # reference — never measured from training data, just a hand-picked
    # threshold, which turned out too tight for real bodies on two exercises
    # live at the gym). Ported here so it can finally get a real measured
    # range the same way every other joint does, via write_exercise_angles.
    for side in ('right', 'left'):
        shoulder, hip = f'{side}_shoulder', f'{side}_hip'
        if vis.get(shoulder, 0.0) >= MIN_VISIBILITY_SPINE and vis.get(hip, 0.0) >= MIN_VISIBILITY_SPINE:
            hx, hy = lm[hip][0], lm[hip][1]
            vertical = (hx, hy - SPINE_UP_DELTA, 0.0)
            result['spine'] = _angle_deg(lm[shoulder], lm[hip], vertical)
            break

    return result


def frame_from_live(landmarks: Dict[int, object], width: int, height: int) -> PoseFrame:
    """
    Adapt the live detector's output (index → Point in pixel coords) to a
    PoseFrame, so the live pipeline computes angles with the exact same math
    and coordinate space (normalized 0-1) used to train the phase models.
    """
    lms: Dict[str, tuple] = {}
    vis: Dict[str, float] = {}
    for idx, name in LANDMARK_NAMES.items():
        pt = landmarks.get(idx)
        if pt is None:
            continue
        lms[name] = (pt.x / width, pt.y / height, getattr(pt, 'z', 0.0) / width)
        vis[name] = getattr(pt, 'visibility', 1.0)
    return PoseFrame(frame_idx=-1, timestamp_ms=0.0, landmarks=lms, visibility=vis)


def angles_over_time(frames: List[PoseFrame]) -> Dict[str, List[float]]:
    """
    Compute angles for every frame and pivot to {joint: [value_per_frame]}.
    Only joints visible in ALL frames are returned (avoids ragged arrays).
    """
    per_frame = [compute_angles(f) for f in frames]

    # Intersect keys so every joint has the same number of samples
    if not per_frame:
        return {}
    common = set(per_frame[0].keys())
    for d in per_frame[1:]:
        common &= d.keys()

    return {joint: [d[joint] for d in per_frame] for joint in sorted(common)}
