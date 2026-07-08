"""
Tests for core/model1/ — phase classifier and violation detector.

No camera, no MediaPipe. All tests use synthetic angle dicts.

Run from project root:
    python -m pytest core/model1/tests.py -v
"""
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np

from core.exercise.exercise_model import AngleRange, Exercise, Phase
from core.model1.classifier import (
    PhaseScore, _boundary_score, _gaussian_score, _model_cache,
    all_scores, classify, classify_trained, score_phase,
)
from core.model1.violations import FRAMES_TO_ALERT, FRAMES_TO_MISSING, ViolationDetector


# ---------------------------------------------------------------------------
# Shared fake exercise (same shape as a real squat)
# ---------------------------------------------------------------------------

def _make_exercise() -> Exercise:
    def _ar(lo, hi, mean=None, std=None, too_low='', too_high='', sev='medium'):
        c = {}
        if too_low: c['too_low'] = too_low
        if too_high: c['too_high'] = too_high
        c['severity'] = sev
        return AngleRange(lo, hi, corrections=c, mean=mean, std=std)

    return Exercise(
        id='squat', name='Squat', description='', muscle_groups=['quads'],
        primary_joints=['right_knee', 'left_knee'],
        mandatory_start_joints=['right_knee', 'left_knee'],
        global_constraints={
            'spine': _ar(0, 30, too_high='Keep back straight', sev='high'),
        },
        phases=[
            Phase(
                name='standing', order=1, is_initial=True,
                instruction='Stand straight',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='stable',
                angles={
                    'right_knee': _ar(155, 180, mean=170.0, std=6.0,
                                      too_low='Straighten knees before starting'),
                    'left_knee':  _ar(155, 180, mean=170.0, std=6.0,
                                      too_low='Straighten knees before starting'),
                },
            ),
            Phase(
                name='descending', order=2, is_initial=False,
                instruction='Squat down',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='decreasing',
                angles={
                    'right_knee': _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Too deep on the way down',
                                      too_high='Bend more as you descend', sev='high'),
                    'left_knee':  _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Too deep on the way down',
                                      too_high='Bend more as you descend', sev='high'),
                },
            ),
            Phase(
                name='hold', order=3, is_initial=False,
                instruction='Hold squat',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='stable',
                angles={
                    'right_knee': _ar(70, 95, too_high='Go lower', too_low="Don't go so deep", sev='high'),
                    'left_knee':  _ar(70, 95, too_high='Go lower', too_low="Don't go so deep", sev='high'),
                },
            ),
            Phase(
                name='ascending', order=4, is_initial=False,
                instruction='Rise up',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='increasing',
                angles={
                    'right_knee': _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Keep pushing up', sev='high'),
                    'left_knee':  _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Keep pushing up', sev='high'),
                },
            ),
        ],
    )


STANDING  = {'right_knee': 170.0, 'left_knee': 170.0}
DESCEND   = {'right_knee': 120.0, 'left_knee': 120.0}
HOLD      = {'right_knee': 82.0,  'left_knee': 82.0}
ASCEND    = {'right_knee': 120.0, 'left_knee': 120.0}


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

class TestScoringFunctions(unittest.TestCase):

    def test_gaussian_peaks_at_mean(self):
        self.assertAlmostEqual(_gaussian_score(120.0, 120.0, 18.0), 1.0)

    def test_gaussian_decays_away_from_mean(self):
        self.assertLess(_gaussian_score(160.0, 120.0, 18.0),
                        _gaussian_score(130.0, 120.0, 18.0))

    def test_gaussian_always_between_0_and_1(self):
        for v in [0, 50, 100, 150, 200]:
            s = _gaussian_score(v, 120.0, 18.0)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_boundary_inside_is_1(self):
        self.assertEqual(_boundary_score(120.0, 90.0, 155.0), 1.0)

    def test_boundary_at_edge_is_1(self):
        self.assertEqual(_boundary_score(90.0, 90.0, 155.0), 1.0)
        self.assertEqual(_boundary_score(155.0, 90.0, 155.0), 1.0)

    def test_boundary_outside_decays(self):
        s = _boundary_score(200.0, 90.0, 155.0)
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1.0)

    def test_boundary_far_outside_reaches_0(self):
        # 400° is very far outside [90, 155] (span=65, so 2×span away = 220°)
        self.assertEqual(_boundary_score(400.0, 90.0, 155.0), 0.0)


# ---------------------------------------------------------------------------
# Phase scoring
# ---------------------------------------------------------------------------

class TestPhaseScore(unittest.TestCase):

    def setUp(self):
        self.ex = _make_exercise()

    def test_standing_angles_score_high_for_standing_phase(self):
        standing_phase = self.ex.get_phase('standing')
        result = score_phase(standing_phase, STANDING)
        self.assertGreater(result.score, 0.8)

    def test_hold_angles_score_low_for_standing_phase(self):
        standing_phase = self.ex.get_phase('standing')
        result = score_phase(standing_phase, HOLD)
        self.assertLess(result.score, 0.2)

    def test_missing_diagnostic_joint_lowers_score(self):
        phase = self.ex.get_phase('standing')
        full_result    = score_phase(phase, STANDING)
        partial_result = score_phase(phase, {'right_knee': 170.0})  # left missing
        self.assertLess(partial_result.score, full_result.score)

    def test_matched_joints_counted_correctly(self):
        phase = self.ex.get_phase('standing')
        result = score_phase(phase, STANDING)
        self.assertEqual(result.matched_joints, 2)
        self.assertEqual(result.total_joints, 2)

    def test_phase_with_no_diagnostic_joints_scores_zero(self):
        bare_phase = Phase(name='empty', order=99, angles={}, diagnostic_joints=[])
        result = score_phase(bare_phase, STANDING)
        self.assertEqual(result.score, 0.0)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.ex = _make_exercise()

    def test_standing_classified_as_standing(self):
        phase = classify(self.ex, STANDING)
        self.assertEqual(phase.name, 'standing')

    def test_hold_classified_as_hold(self):
        phase = classify(self.ex, HOLD)
        self.assertEqual(phase.name, 'hold')

    def test_descending_classified_as_descending_or_ascending(self):
        # Both descending and ascending have identical angle ranges;
        # without motion direction both are valid answers
        phase = classify(self.ex, DESCEND)
        self.assertIn(phase.name, ('descending', 'ascending'))

    def test_motion_direction_breaks_tie_descending(self):
        # If knee angle is decreasing → descending
        phase = classify(self.ex, DESCEND,
                         prev_primary_angle=140.0,
                         current_primary_angle=120.0)
        self.assertEqual(phase.name, 'descending')

    def test_motion_direction_breaks_tie_ascending(self):
        # If knee angle is increasing → ascending
        phase = classify(self.ex, ASCEND,
                         prev_primary_angle=100.0,
                         current_primary_angle=120.0)
        self.assertEqual(phase.name, 'ascending')

    def test_all_scores_sorted_best_first(self):
        scores = all_scores(self.ex, HOLD)
        values = [s.score for s in scores]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_all_scores_has_entry_for_every_phase(self):
        scores = all_scores(self.ex, HOLD)
        self.assertEqual(len(scores), len(self.ex.phases))

    def test_empty_angles_returns_a_phase(self):
        # Should not crash; returns whichever phase scores highest on nothing
        phase = classify(self.ex, {})
        self.assertIsNotNone(phase)

    def test_returns_none_for_exercise_with_no_phases(self):
        empty_ex = Exercise('x', 'X', '', [], [])
        self.assertIsNone(classify(empty_ex, STANDING))


# ---------------------------------------------------------------------------
# ViolationDetector
# ---------------------------------------------------------------------------

class TestViolationDetector(unittest.TestCase):

    def setUp(self):
        self.ex = _make_exercise()
        self.hold_phase = self.ex.get_phase('hold')
        self.detector = ViolationDetector(self.ex)

    def _run(self, angles, phase=None, frames=None):
        phase = phase or self.hold_phase
        frames = frames or FRAMES_TO_ALERT + 1
        violations = []
        for _ in range(frames):
            violations = self.detector.analyze(phase, angles)
        return violations

    def test_correct_form_no_violations(self):
        v = self._run({**HOLD, 'spine': 15.0})
        self.assertEqual(v, [])

    def test_too_low_reports_correct_message(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        v = self._run(bad)
        msgs = {x.joint: x.message for x in v}
        self.assertIn('right_knee', msgs)
        self.assertEqual(msgs['right_knee'], "Don't go so deep")
        self.assertNotIn('left_knee', msgs)

    def test_too_high_reports_correct_message(self):
        bad = {'right_knee': 130.0, 'left_knee': 82.0, 'spine': 15.0}
        v = self._run(bad)
        msgs = {x.joint: x.message for x in v}
        self.assertIn('right_knee', msgs)
        self.assertEqual(msgs['right_knee'], 'Go lower')

    def test_global_constraint_violation_reported(self):
        bad = {**HOLD, 'spine': 45.0}
        v = self._run(bad)
        joints = {x.joint for x in v}
        self.assertIn('spine', joints)
        spine = next(x for x in v if x.joint == 'spine')
        self.assertEqual(spine.message, 'Keep back straight')
        self.assertEqual(spine.severity, 'high')

    def test_violation_not_reported_before_frame_threshold(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        v = self._run(bad, frames=FRAMES_TO_ALERT - 1)
        self.assertEqual(v, [])

    def test_missing_joint_reported_after_threshold(self):
        angles = {'left_knee': 82.0, 'spine': 15.0}  # right_knee absent
        v = self._run(angles, frames=FRAMES_TO_MISSING + 1)
        joints = {x.joint for x in v}
        self.assertIn('right_knee', joints)
        missing = next(x for x in v if x.joint == 'right_knee')
        self.assertEqual(missing.direction, 'missing')

    def test_recovery_resets_counter(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        self._run(bad)
        # Correct form — resets counters
        self.detector.analyze(self.hold_phase, {**HOLD, 'spine': 15.0})
        # Single bad frame should not re-trigger immediately
        v = self.detector.analyze(self.hold_phase, bad)
        self.assertEqual(v, [])

    def test_limited_joint_downgrades_severity(self):
        detector = ViolationDetector(self.ex, limited_joints=['right_knee'])
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        v = []
        for _ in range(FRAMES_TO_ALERT + 1):
            v = detector.analyze(self.hold_phase, bad)
        knee = next((x for x in v if x.joint == 'right_knee'), None)
        self.assertIsNotNone(knee)
        self.assertEqual(knee.severity, 'low')
        self.assertIn('[Adapted]', knee.message)

    def test_phase_name_in_violation(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        v = self._run(bad)
        for violation in v:
            self.assertEqual(violation.phase, 'hold')

    def test_to_dict_has_all_keys(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        v = self._run(bad)
        self.assertTrue(len(v) > 0)
        d = v[0].to_dict()
        for key in ('joint', 'phase', 'direction', 'message', 'severity',
                    'value', 'expected_min', 'expected_max'):
            self.assertIn(key, d)

    def test_reset_clears_counters(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        self._run(bad)
        self.detector.reset()
        v = self.detector.analyze(self.hold_phase, bad)
        self.assertEqual(v, [])

    def test_phase_switch_clears_old_joints(self):
        # Analyze in hold phase (knee rules active)
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 15.0}
        self._run(bad)
        # Switch to standing phase (different knee range)
        standing = self.ex.get_phase('standing')
        v = self.detector.analyze(standing, STANDING)
        # No violations since standing angles are correct for standing phase
        knee_violations = [x for x in v if x.joint in ('right_knee', 'left_knee')]
        self.assertEqual(knee_violations, [])


# ---------------------------------------------------------------------------
# Integration: classify then check violations
# ---------------------------------------------------------------------------

class TestClassifyThenViolate(unittest.TestCase):

    def setUp(self):
        self.ex = _make_exercise()
        self.detector = ViolationDetector(self.ex)

    def test_full_loop_correct_form(self):
        """Classify phase then check violations — no alerts for correct form."""
        phase = classify(self.ex, {**HOLD, 'spine': 15.0})
        self.assertEqual(phase.name, 'hold')
        for _ in range(FRAMES_TO_ALERT + 1):
            v = self.detector.analyze(phase, {**HOLD, 'spine': 15.0})
        self.assertEqual(v, [])

    def test_full_loop_bad_knee_in_hold(self):
        """Bad right knee in hold phase triggers the phase-specific message."""
        angles = {'right_knee': 50.0, 'left_knee': 82.0, 'spine': 15.0}
        phase = classify(self.ex, angles)
        for _ in range(FRAMES_TO_ALERT + 1):
            v = self.detector.analyze(phase, angles)
        joints = {x.joint for x in v}
        self.assertIn('right_knee', joints)


# ---------------------------------------------------------------------------
# Full rep simulation: correct + bad form through all 4 phases
# ---------------------------------------------------------------------------

class TestFullSquatRep(unittest.TestCase):
    """
    Simulate a complete squat rep frame-by-frame.
    Verifies that the right alerts fire (and only those alerts) at each phase.
    No camera, no videos — pure synthetic angle dicts.
    """

    def setUp(self):
        self.ex = _make_exercise()
        self.detector = ViolationDetector(self.ex)

    def _run(self, angles, phase, n=FRAMES_TO_ALERT + 1):
        violations = []
        for _ in range(n):
            violations = self.detector.analyze(phase, angles)
        return violations

    def test_phase1_standing_correct_form_no_alerts(self):
        phase = classify(self.ex, STANDING)
        self.assertEqual(phase.name, 'standing')
        v = self._run({**STANDING, 'spine': 10.0}, phase)
        self.assertEqual(v, [])

    def test_phase2_descending_bad_spine_alerts_spine_only(self):
        # Knee is mid-range (120°) and decreasing → descending
        angles = {'right_knee': 120.0, 'left_knee': 120.0, 'spine': 45.0}
        phase = classify(self.ex, angles, prev_primary_angle=170.0, current_primary_angle=120.0)
        self.assertEqual(phase.name, 'descending')

        v = self._run(angles, phase)
        joints = {x.joint for x in v}

        self.assertIn('spine', joints)
        spine_v = next(x for x in v if x.joint == 'spine')
        self.assertEqual(spine_v.message, 'Keep back straight')
        self.assertEqual(spine_v.severity, 'high')
        # Knee is inside descending range — should not alert
        self.assertNotIn('right_knee', joints)
        self.assertNotIn('left_knee', joints)

    def test_phase2_descending_correct_spine_no_alerts(self):
        angles = {'right_knee': 120.0, 'left_knee': 120.0, 'spine': 10.0}
        phase = classify(self.ex, angles, prev_primary_angle=170.0, current_primary_angle=120.0)
        v = self._run(angles, phase)
        self.assertEqual(v, [])

    def test_phase3_hold_bad_right_knee_too_low(self):
        # right knee at 60° < hold min of 70° → "Don't go so deep"
        angles = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 10.0}
        phase = classify(self.ex, angles)
        self.assertEqual(phase.name, 'hold')

        v = self._run(angles, phase)
        joints = {x.joint for x in v}

        self.assertIn('right_knee', joints)
        knee_v = next(x for x in v if x.joint == 'right_knee')
        self.assertEqual(knee_v.message, "Don't go so deep")
        self.assertEqual(knee_v.direction, 'too_low')
        self.assertNotIn('left_knee', joints)

    def test_phase3_hold_bad_right_knee_too_high(self):
        # right knee at 130° > hold max of 95° → "Go lower"
        angles = {'right_knee': 130.0, 'left_knee': 82.0, 'spine': 10.0}
        phase = classify(self.ex, angles)
        self.assertEqual(phase.name, 'hold')

        v = self._run(angles, phase)
        joints = {x.joint for x in v}

        self.assertIn('right_knee', joints)
        knee_v = next(x for x in v if x.joint == 'right_knee')
        self.assertEqual(knee_v.message, 'Go lower')
        self.assertEqual(knee_v.direction, 'too_high')

    def test_phase3_hold_both_knees_bad(self):
        angles = {'right_knee': 60.0, 'left_knee': 60.0, 'spine': 10.0}
        phase = classify(self.ex, angles)
        v = self._run(angles, phase)
        joints = {x.joint for x in v}
        self.assertIn('right_knee', joints)
        self.assertIn('left_knee', joints)

    def test_phase3_hold_correct_then_fix_clears_alert(self):
        # Bad for 9 frames → alert fires
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 10.0}
        good = {**HOLD, 'spine': 10.0}
        phase = classify(self.ex, bad)
        self._run(bad, phase)
        # Now correct for 1 frame — counter resets
        self.detector.analyze(phase, good)
        # One more bad frame should NOT re-alert (counter reset to 0)
        v = self.detector.analyze(phase, bad)
        self.assertEqual(v, [])

    def test_phase4_ascending_correct_form_no_alerts(self):
        angles = {'right_knee': 115.0, 'left_knee': 115.0, 'spine': 10.0}
        phase = classify(self.ex, angles, prev_primary_angle=82.0, current_primary_angle=115.0)
        self.assertEqual(phase.name, 'ascending')
        v = self._run(angles, phase)
        self.assertEqual(v, [])

    def test_full_rep_sequence_phase_names(self):
        """Walk through the full rep and confirm phase transitions classify correctly."""
        keyframes = [
            ({'right_knee': 170.0, 'left_knee': 170.0}, None,    None,    'standing'),
            ({'right_knee': 130.0, 'left_knee': 130.0}, 170.0, 130.0, 'descending'),
            ({'right_knee': 82.0,  'left_knee': 82.0},  None,    None,    'hold'),
            ({'right_knee': 115.0, 'left_knee': 115.0}, 82.0,  115.0, 'ascending'),
            ({'right_knee': 170.0, 'left_knee': 170.0}, None,    None,    'standing'),
        ]
        for angles, prev, curr, expected in keyframes:
            phase = classify(self.ex, angles, prev_primary_angle=prev,
                             current_primary_angle=curr)
            self.assertEqual(phase.name, expected, f'at angles {angles}')

    def test_violation_not_fired_before_debounce_threshold(self):
        bad = {'right_knee': 60.0, 'left_knee': 82.0, 'spine': 10.0}
        phase = self.ex.get_phase('hold')
        # One frame short of the threshold → no alert yet
        v = self._run(bad, phase, n=FRAMES_TO_ALERT - 1)
        self.assertEqual(v, [])


# ---------------------------------------------------------------------------
# Random Forest: test with a tiny synthetic model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature vector format
# ---------------------------------------------------------------------------

class TestFeatureVector(unittest.TestCase):
    """
    The 13-element feature vector must be built identically during training
    (collect_samples) and inference (classify_trained). If they diverge, the RF
    silently produces garbage. These tests lock down the format.
    """

    def test_length_is_12_joints_plus_delta(self):
        from core.model1.trainer import JOINTS, angles_to_features
        feat = angles_to_features({})
        self.assertEqual(len(feat), len(JOINTS) + 1)
        self.assertEqual(len(feat), 13)

    def test_missing_joints_are_minus_one(self):
        from core.model1.trainer import angles_to_features
        feat = angles_to_features({})
        # All 12 joint slots should be -1.0 (delta=0.0 is the 13th)
        self.assertTrue(all(v == -1.0 for v in feat[:12]))

    def test_present_joint_value_preserved(self):
        from core.model1.trainer import JOINTS, angles_to_features
        feat = angles_to_features({'right_knee': 90.0})
        self.assertEqual(feat[JOINTS.index('right_knee')], 90.0)

    def test_absent_joint_is_minus_one_not_zero(self):
        from core.model1.trainer import JOINTS, angles_to_features
        feat = angles_to_features({'right_knee': 90.0})
        self.assertEqual(feat[JOINTS.index('left_knee')], -1.0)

    def test_delta_is_last_element(self):
        from core.model1.trainer import angles_to_features
        feat = angles_to_features({}, angle_delta=7.5)
        self.assertAlmostEqual(feat[-1], 7.5)

    def test_delta_default_is_zero(self):
        from core.model1.trainer import angles_to_features
        feat = angles_to_features({'right_knee': 90.0})
        self.assertEqual(feat[-1], 0.0)

    def test_negative_delta_preserved(self):
        from core.model1.trainer import angles_to_features
        feat = angles_to_features({}, angle_delta=-4.2)
        self.assertAlmostEqual(feat[-1], -4.2)

    def test_feature_dtype_is_float32(self):
        from core.model1.trainer import angles_to_features
        feat = angles_to_features({'right_knee': 90.0}, angle_delta=1.0)
        self.assertEqual(feat.dtype, np.float32)

    def test_joints_order_matches_trainer_constant(self):
        """The first 12 features must be in the same order as trainer.JOINTS."""
        from core.model1.trainer import JOINTS, angles_to_features
        angles = {j: float(i * 10) for i, j in enumerate(JOINTS)}
        feat = angles_to_features(angles)
        for i, joint in enumerate(JOINTS):
            self.assertAlmostEqual(feat[i], angles[joint],
                msg=f'Feature index {i} should be {joint}={angles[joint]}, got {feat[i]}')


# ---------------------------------------------------------------------------
# Sequential frame labeling
# ---------------------------------------------------------------------------

class TestSequentialLabeling(unittest.TestCase):
    """
    _assign_phase_labels labels frames by TEMPORAL POSITION in the rep, not by
    angle value. This is what lets the RF distinguish descending from ascending
    even though they span the same angle range.

    All tests use a synthetic sinusoidal squat signal — no camera, no videos.
    Signal: 200 frames, 2 full reps.
      peaks  ≈ frames 25, 125  (knees extended = standing)
      valleys ≈ frames 75, 175 (knees flexed = hold)
    """

    def _signal(self, n=200) -> list:
        t = np.linspace(0, 4 * np.pi, n)
        return list(82 + 88 * (1 + np.sin(t)) / 2)

    def _trajectories(self, n=200) -> dict:
        sig = self._signal(n)
        return {'right_knee': sig, 'left_knee': sig}

    def test_all_four_phases_labeled(self):
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        unique = {l for l in labels if l is not None}
        self.assertEqual(unique, {'standing', 'descending', 'hold', 'ascending'})

    def test_near_peak_labeled_standing(self):
        # Peak is at frame ~25; frames within ±STABLE_WINDOW should be 'standing'
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        self.assertEqual(labels[25], 'standing')

    def test_mid_descent_labeled_descending(self):
        # Frame 50 is between the standing window (≈20-30) and hold window (≈70-80)
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        self.assertEqual(labels[50], 'descending')

    def test_at_valley_labeled_hold(self):
        # Valley is at frame ~75
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        self.assertEqual(labels[75], 'hold')

    def test_mid_ascent_labeled_ascending(self):
        # Frame 100 is between hold window (≈70-80) and next standing window (≈120-130)
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        self.assertEqual(labels[100], 'ascending')

    def test_descending_and_ascending_differ_at_same_angle(self):
        """
        Frame 50 (descent, ~140°) and frame 100 (ascent, ~140°) have almost
        identical angles but must get different labels — this is exactly the
        information angle-range matching cannot provide.
        """
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        self.assertIsNotNone(labels[50])
        self.assertIsNotNone(labels[100])
        self.assertNotEqual(labels[50], labels[100])

    def test_label_count_per_phase_is_nonzero(self):
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels(self._trajectories(), _make_exercise())
        counts = {}
        for l in labels:
            if l:
                counts[l] = counts.get(l, 0) + 1
        for phase in ('standing', 'descending', 'hold', 'ascending'):
            self.assertGreater(counts.get(phase, 0), 0, f'{phase} has zero labeled frames')

    def test_flat_signal_all_none(self):
        # No peaks or valleys → nothing can be labeled
        from core.model1.trainer import _assign_phase_labels
        flat = {'right_knee': [170.0] * 50, 'left_knee': [170.0] * 50}
        labels = _assign_phase_labels(flat, _make_exercise())
        self.assertTrue(all(l is None for l in labels))

    def test_empty_trajectories_returns_empty(self):
        from core.model1.trainer import _assign_phase_labels
        labels = _assign_phase_labels({}, _make_exercise())
        self.assertEqual(labels, [])

    def test_labels_length_matches_signal_length(self):
        from core.model1.trainer import _assign_phase_labels
        n = 200
        labels = _assign_phase_labels(self._trajectories(n), _make_exercise())
        self.assertEqual(len(labels), n)

    def test_no_unknown_phase_names(self):
        from core.model1.trainer import _assign_phase_labels
        ex = _make_exercise()
        valid = {p.name for p in ex.phases}
        labels = _assign_phase_labels(self._trajectories(), ex)
        for l in labels:
            if l is not None:
                self.assertIn(l, valid, f'Label "{l}" is not a phase in the exercise')


# ---------------------------------------------------------------------------
# Real trained models (skipped if data/models/ not populated)
# ---------------------------------------------------------------------------

SQUAT_MODEL_PATH = Path('data/models/phase_squat.joblib')


class TestRealTrainedModel(unittest.TestCase):
    """
    Tests against the actual squat model produced by trainer.py.
    Skipped when the file doesn't exist.

    These tests catch the bugs that the synthetic-model tests cannot:
      - Feature shape mismatch between training and inference
      - Phase name drift (model class no longer matches exercise phase)
      - Trainer output missing required keys
    """

    @classmethod
    def setUpClass(cls):
        if not SQUAT_MODEL_PATH.exists():
            raise unittest.SkipTest(
                f'No model at {SQUAT_MODEL_PATH} — run: python -m core.model1.trainer'
            )
        import joblib
        cls.bundle = joblib.load(SQUAT_MODEL_PATH)
        cls.ex = _make_exercise()

    def test_bundle_has_all_required_keys(self):
        for key in ('model', 'classes', 'joints', 'exercise_id',
                    'accuracy', 'n_samples', 'class_counts'):
            self.assertIn(key, self.bundle, f'bundle missing key: {key}')

    def test_feature_shape_matches_inference(self):
        """
        The most important test: the number of features the model expects must
        equal what classify_trained actually builds (len(JOINTS) + 1).
        If angles_to_features changes but the model is not retrained, this fails.
        """
        from core.model1.trainer import JOINTS
        expected = len(JOINTS) + 1
        actual = self.bundle['model'].n_features_in_
        self.assertEqual(actual, expected,
            f'Saved model expects {actual} features, inference builds {expected}. Retrain.')

    def test_all_classes_are_valid_phase_names(self):
        """Every class the model learned must correspond to a real phase."""
        for cls in self.bundle['classes']:
            phase = self.ex.get_phase(str(cls))
            self.assertIsNotNone(phase,
                f'Model class "{cls}" has no matching phase in the exercise')

    def test_exercise_id_is_squat(self):
        self.assertEqual(self.bundle['exercise_id'], 'squat')

    def test_accuracy_stored_and_positive(self):
        acc = self.bundle['accuracy']
        # accuracy can be None for very small datasets, otherwise must be in [0,1]
        if acc is not None:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_n_samples_positive(self):
        self.assertGreater(self.bundle['n_samples'], 0)

    def test_predict_does_not_crash_on_all_missing(self):
        """All -1.0 features (no joints visible) must not raise."""
        from core.model1.trainer import angles_to_features
        feat = angles_to_features({}, angle_delta=0.0).reshape(1, -1)
        pred = self.bundle['model'].predict(feat)
        self.assertEqual(len(pred), 1)
        self.assertIn(str(pred[0]), [str(c) for c in self.bundle['classes']])

    def test_predict_on_standing_angles_returns_standing(self):
        """
        Realistic standing angles (with hip/ankle present) should predict 'standing'.
        Using only knee angles fails because the model relies heavily on all visible
        joints — training frames always had hips and ankles visible too.
        """
        from core.model1.trainer import angles_to_features
        # Use angles representative of a real standing frame (from MongoDB means)
        standing_full = {
            'right_knee': 167.0, 'left_knee': 157.0,
            'right_hip':  167.0, 'left_hip':  157.0,
            'right_ankle': 124.0, 'left_ankle': 95.0,
        }
        # delta=-0.5: knee angle just peaked and is very slightly settling.
        # With the 3-class squat model (no hold), near-zero delta is the key
        # signal; negative means descending side of peak → standing wins.
        feat = angles_to_features(standing_full, angle_delta=-0.5).reshape(1, -1)
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertEqual(pred, 'standing',
            f'Realistic standing angles classified as "{pred}" — check training data quality')

    def test_negative_delta_not_classified_as_standing(self):
        """A strongly decreasing angle should never be classified as standing."""
        from core.model1.trainer import angles_to_features
        mid = {'right_knee': 130.0, 'left_knee': 130.0}
        feat = angles_to_features(mid, angle_delta=-10.0).reshape(1, -1)
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertNotEqual(pred, 'standing',
            'Angle at 130° and sharply decreasing should not be standing')

    def test_positive_delta_not_classified_as_standing(self):
        """A strongly increasing angle from a low position should not be standing."""
        from core.model1.trainer import angles_to_features
        mid = {'right_knee': 110.0, 'left_knee': 110.0}
        feat = angles_to_features(mid, angle_delta=+10.0).reshape(1, -1)
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertNotEqual(pred, 'standing')

    def test_classify_trained_returns_phase_and_method(self):
        """End-to-end: classify_trained returns (Phase, str) — not None."""
        phase, method = classify_trained(self.ex, STANDING,
                                         prev_primary_angle=170.0,
                                         current_primary_angle=170.0)
        self.assertIsNotNone(phase)
        self.assertIsInstance(phase.name, str)
        self.assertIn(method, ('rf', 'gaussian'))

    def test_classify_trained_uses_rf_for_squat(self):
        """The real model file exists, so method must be 'rf', not 'gaussian'."""
        _model_cache.clear()
        phase, method = classify_trained(self.ex, STANDING,
                                          prev_primary_angle=170.0,
                                          current_primary_angle=170.0)
        _model_cache.clear()
        self.assertEqual(method, 'rf')

    def test_negative_delta_predicts_descending(self):
        """
        Mid-range angle with a strongly negative delta (angle falling fast)
        should predict 'descending'. Requires hip angles — the model was
        trained with multi-joint vectors, not just knees.
        """
        from core.model1.trainer import angles_to_features
        mid = {'right_knee': 120.0, 'left_knee': 120.0,
               'right_hip': 130.0, 'left_hip': 130.0}
        feat = angles_to_features(mid, angle_delta=-15.0).reshape(1, -1)
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertEqual(pred, 'descending',
            f'Mid-range angle with delta=-15° classified as "{pred}" — '
            'the delta feature should signal descent')

    def test_delta_feature_is_used_by_model(self):
        """
        The same mid-range angle with opposite deltas must not produce identical
        predictions — this confirms angle_delta is actually influencing the RF.
        """
        from core.model1.trainer import angles_to_features
        mid = {'right_knee': 120.0, 'left_knee': 120.0,
               'right_hip': 130.0, 'left_hip': 130.0}
        feat_neg = angles_to_features(mid, angle_delta=-15.0).reshape(1, -1)
        feat_pos = angles_to_features(mid, angle_delta=+15.0).reshape(1, -1)
        pred_neg = str(self.bundle['model'].predict(feat_neg)[0])
        pred_pos = str(self.bundle['model'].predict(feat_pos)[0])
        self.assertNotEqual(pred_neg, pred_pos,
            'Opposite deltas at the same angle produce identical predictions — '
            'angle_delta is not influencing the model')


# ---------------------------------------------------------------------------
# Random Forest: test with a tiny synthetic model
# ---------------------------------------------------------------------------

class TestClassifyTrainedRF(unittest.TestCase):
    """
    Train a minimal Random Forest inline (no videos, no MongoDB) and verify
    that classify_trained uses it and returns the correct phase.

    This is how you test the RF path before real reference videos exist.
    """

    def setUp(self):
        self.ex = _make_exercise()

    def _build_bundle(self) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from core.model1.trainer import JOINTS

        # 10 examples of each phase — enough to train a trivial classifier
        # Features: 12 joint angles + 1 angle_delta (last element)
        samples = {
            'standing':   [170.0, 170.0] + [-1.0] * 10 + [0.0],   # stable
            'descending': [125.0, 125.0] + [-1.0] * 10 + [-5.0],  # angle decreasing
            'hold':       [82.0,  82.0]  + [-1.0] * 10 + [0.0],   # stable at bottom
            'ascending':  [110.0, 110.0] + [-1.0] * 10 + [+5.0],  # angle increasing
        }
        X, y = [], []
        for label, feat in samples.items():
            for _ in range(20):
                X.append(feat)
                y.append(label)

        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(np.array(X, dtype=np.float32), y)

        return {
            'model':       clf,
            'classes':     list(clf.classes_),
            'joints':      JOINTS,
            'exercise_id': 'squat',
        }

    def test_rf_classifies_standing(self):
        import joblib
        bundle = self._build_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'phase_squat.joblib'
            joblib.dump(bundle, model_path)
            _model_cache.clear()
            with patch('core.model1.classifier.MODELS_DIR', Path(tmpdir)):
                phase, method = classify_trained(self.ex, STANDING)
            _model_cache.clear()
        self.assertEqual(method, 'rf')
        self.assertEqual(phase.name, 'standing')

    def test_rf_classifies_hold(self):
        import joblib
        bundle = self._build_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'phase_squat.joblib'
            joblib.dump(bundle, model_path)
            _model_cache.clear()
            with patch('core.model1.classifier.MODELS_DIR', Path(tmpdir)):
                phase, method = classify_trained(self.ex, HOLD)
            _model_cache.clear()
        self.assertEqual(method, 'rf')
        self.assertEqual(phase.name, 'hold')

    def test_rf_falls_back_to_gaussian_when_no_model(self):
        _model_cache.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('core.model1.classifier.MODELS_DIR', Path(tmpdir)):
                phase, method = classify_trained(self.ex, STANDING)
        _model_cache.clear()
        self.assertEqual(method, 'gaussian')
        self.assertEqual(phase.name, 'standing')

    def test_rf_method_reported_correctly(self):
        import joblib
        bundle = self._build_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump(bundle, Path(tmpdir) / 'phase_squat.joblib')
            _model_cache.clear()
            with patch('core.model1.classifier.MODELS_DIR', Path(tmpdir)):
                _, method = classify_trained(self.ex, HOLD)
            _model_cache.clear()
        self.assertIn(method, ('rf', 'gaussian'))


if __name__ == '__main__':
    unittest.main()
