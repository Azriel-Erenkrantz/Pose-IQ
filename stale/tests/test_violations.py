"""
Tests for core/ml/violations.py — never wired into the live pipeline or the
web app; only ever exercised by its own tests. Split out of tests/test_ml.py
when violations.py moved to stale/.

No camera, no MediaPipe. All tests use synthetic angle dicts.

Run from project root:
    python -m pytest stale/tests/test_violations.py -v
"""
import unittest

from core.exercise.exercise_model import AngleRange, Exercise, Phase
from core.ml.classifier import classify
from stale.core.ml.violations import FRAMES_TO_ALERT, FRAMES_TO_MISSING, ViolationDetector


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


if __name__ == '__main__':
    unittest.main()
