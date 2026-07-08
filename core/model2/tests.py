"""
Unit tests for core/model2/ — the video-to-JSON pipeline.

Tests use synthetic data (no real videos or MediaPipe calls needed).
"""
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from .angles import ANGLE_DEFS, MIN_VISIBILITY, compute_angles
from .extractor import PoseFrame
from .phases import Phase, segment_reps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(landmarks: dict, visibility: dict | None = None, frame_idx: int = 0) -> PoseFrame:
    vis = {k: 1.0 for k in landmarks} if visibility is None else visibility
    return PoseFrame(frame_idx=frame_idx, timestamp_ms=frame_idx * 33.3, landmarks=landmarks, visibility=vis)


def _right_angle_landmarks():
    """Landmarks that produce a 90° right_knee angle."""
    return {
        'right_hip':   (0.5, 0.3, 0.0),
        'right_knee':  (0.5, 0.6, 0.0),  # vertex
        'right_ankle': (0.8, 0.6, 0.0),  # arm goes right → 90°
        # others needed for other angle defs (but low visibility → skipped)
    }


def _straight_leg_landmarks():
    """Landmarks that produce a ~180° right_knee angle (straight leg)."""
    return {
        'right_hip':   (0.5, 0.3, 0.0),
        'right_knee':  (0.5, 0.6, 0.0),
        'right_ankle': (0.5, 0.9, 0.0),  # perfectly collinear
    }


def _sine_trajectory(n_frames: int = 120, cycles: int = 3, amplitude: float = 40.0, offset: float = 120.0):
    """Simulate a joint going up and down — like squatting and standing."""
    t = np.linspace(0, cycles * 2 * math.pi, n_frames)
    return (offset + amplitude * np.sin(t)).tolist()


# ---------------------------------------------------------------------------
# angles.py tests
# ---------------------------------------------------------------------------

class TestComputeAngles(unittest.TestCase):

    def test_right_angle_returns_90(self):
        lm = _right_angle_landmarks()
        vis = {k: 1.0 for k in lm}
        frame = _make_frame(lm, vis)
        angles = compute_angles(frame)
        self.assertIn('right_knee', angles)
        self.assertAlmostEqual(angles['right_knee'], 90.0, delta=0.5)

    def test_straight_leg_returns_180(self):
        lm = _straight_leg_landmarks()
        vis = {k: 1.0 for k in lm}
        frame = _make_frame(lm, vis)
        angles = compute_angles(frame)
        self.assertIn('right_knee', angles)
        self.assertAlmostEqual(angles['right_knee'], 180.0, delta=0.5)

    def test_low_visibility_joint_excluded(self):
        lm = _right_angle_landmarks()
        vis = {k: 1.0 for k in lm}
        vis['right_ankle'] = MIN_VISIBILITY - 0.01  # just below threshold
        frame = _make_frame(lm, vis)
        angles = compute_angles(frame)
        self.assertNotIn('right_knee', angles)

    def test_all_invisible_returns_empty(self):
        lm = {j: (0.5, 0.5, 0.0) for j in ['right_hip', 'right_knee', 'right_ankle']}
        vis = {k: 0.0 for k in lm}
        frame = _make_frame(lm, vis)
        angles = compute_angles(frame)
        self.assertEqual(angles, {})

    def test_angle_between_0_and_180(self):
        """All computed angles must be in [0, 180]."""
        # Build a frame with all required joints at plausible positions
        all_joints = set()
        for a, b, c in ANGLE_DEFS.values():
            all_joints.update([a, b, c])
        rng = np.random.default_rng(42)
        lm  = {j: tuple(rng.random(3).tolist()) for j in all_joints}
        vis = {j: 1.0 for j in all_joints}
        frame = _make_frame(lm, vis)
        angles = compute_angles(frame)
        for name, val in angles.items():
            self.assertGreaterEqual(val, 0.0,   f'{name} angle below 0')
            self.assertLessEqual(val,   180.0,  f'{name} angle above 180')

    def test_symmetric_landmarks_produce_equal_angles(self):
        """Mirror-image landmarks should give the same angle on left and right."""
        lm = {
            'left_hip':    (0.3, 0.3, 0.0),
            'left_knee':   (0.3, 0.6, 0.0),
            'left_ankle':  (0.6, 0.6, 0.0),
            'right_hip':   (0.7, 0.3, 0.0),
            'right_knee':  (0.7, 0.6, 0.0),
            'right_ankle': (0.4, 0.6, 0.0),
        }
        vis = {k: 1.0 for k in lm}
        frame = _make_frame(lm, vis)
        angles = compute_angles(frame)
        self.assertIn('left_knee',  angles)
        self.assertIn('right_knee', angles)
        self.assertAlmostEqual(angles['left_knee'], angles['right_knee'], delta=0.5)


# ---------------------------------------------------------------------------
# phases.py tests
# ---------------------------------------------------------------------------

class TestSegmentReps(unittest.TestCase):

    def test_sine_wave_produces_multiple_reps(self):
        trajectory = {'right_knee': _sine_trajectory(n_frames=180, cycles=3)}
        reps = segment_reps(trajectory)
        # 3 sine cycles → expect 2-4 reps detected
        self.assertGreaterEqual(len(reps), 2)

    def test_each_rep_has_at_least_one_phase(self):
        trajectory = {'right_knee': _sine_trajectory(n_frames=180, cycles=3)}
        reps = segment_reps(trajectory)
        for rep in reps:
            self.assertGreater(len(rep), 0)

    def test_flat_signal_returns_single_phase(self):
        trajectory = {'right_knee': [120.0] * 60}
        reps = segment_reps(trajectory)
        self.assertEqual(len(reps), 1)
        self.assertEqual(len(reps[0]), 1)

    def test_empty_trajectories_returns_empty(self):
        reps = segment_reps({})
        self.assertEqual(reps, [])

    def test_phase_frames_are_contiguous(self):
        trajectory = {'right_knee': _sine_trajectory(n_frames=120, cycles=2)}
        reps = segment_reps(trajectory)
        for rep in reps:
            for i in range(len(rep) - 1):
                self.assertEqual(rep[i].end_frame, rep[i + 1].start_frame)

    def test_phase_stats_keys(self):
        trajectory = {'right_knee': _sine_trajectory()}
        reps = segment_reps(trajectory)
        for rep in reps:
            for phase in rep:
                stats = phase.stats
                for joint, s in stats.items():
                    self.assertIn('min',  s)
                    self.assertIn('max',  s)
                    self.assertIn('mean', s)
                    self.assertIn('std',  s)

    def test_phase_min_leq_max(self):
        trajectory = {'right_knee': _sine_trajectory()}
        reps = segment_reps(trajectory)
        for rep in reps:
            for phase in rep:
                for joint, s in phase.stats.items():
                    self.assertLessEqual(s['min'], s['max'])


# ---------------------------------------------------------------------------
# builder.py — exercise_id normalization
# ---------------------------------------------------------------------------

class TestExerciseIdNormalization(unittest.TestCase):

    def test_spaces_replaced_with_underscores(self):
        from .builder import _exercise_id
        p = Path('/data/videos/biceps curl')
        self.assertEqual(_exercise_id(p), 'biceps_curl')

    def test_hyphens_replaced_with_underscores(self):
        from .builder import _exercise_id
        p = Path('/data/videos/bench-press')
        self.assertEqual(_exercise_id(p), 'bench_press')

    def test_already_clean_unchanged(self):
        from .builder import _exercise_id
        p = Path('/data/videos/squat')
        self.assertEqual(_exercise_id(p), 'squat')

    def test_uppercase_lowercased(self):
        from .builder import _exercise_id
        p = Path('/data/videos/DeadLift')
        self.assertEqual(_exercise_id(p), 'deadlift')


if __name__ == '__main__':
    unittest.main()
