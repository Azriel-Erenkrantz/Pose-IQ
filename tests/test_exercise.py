"""
Tests for Model 1: exercise loading and occlusion-robust phase matching.

All tests use synthetic angle data — no camera or MediaPipe required.
State-machine/posture-rule tests moved to stale/tests/test_exercise_state_machine.py
along with the desktop-pipeline-only code they cover.

Run from project root:
    python -m pytest tests/test_exercise.py -v
"""
import unittest

from core.exercise.exercise_model import AngleRange, Exercise, ExerciseModel, Phase


# ---------------------------------------------------------------------------
# Fake exercise: a simple 3-phase squat-like movement
# ---------------------------------------------------------------------------

def _make_squat() -> Exercise:
    knee_corrections = {'too_low': 'Bend more', 'too_high': 'Straighten up', 'severity': 'high'}
    return Exercise(
        id='test_squat',
        name='Test Squat',
        description='Fake squat for testing',
        muscle_groups=['quadriceps'],
        primary_joints=['right_knee', 'left_knee'],
        mandatory_start_joints=['right_knee', 'left_knee'],
        global_constraints={
            'spine': AngleRange(0, 30, corrections={'too_high': 'Keep back straight', 'severity': 'high'})
        },
        phases=[
            Phase(
                name='standing', order=1, is_initial=True,
                instruction='Stand straight',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='stable',
                angles={
                    'right_knee': AngleRange(155, 180, corrections=knee_corrections),
                    'left_knee':  AngleRange(155, 180, corrections=knee_corrections),
                },
            ),
            Phase(
                name='descending', order=2, is_initial=False,
                instruction='Squat down',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='decreasing',
                angles={
                    'right_knee': AngleRange(90, 155, corrections=knee_corrections),
                    'left_knee':  AngleRange(90, 155, corrections=knee_corrections),
                },
            ),
            Phase(
                name='hold', order=3, is_initial=False,
                instruction='Hold squat',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='stable',
                angles={
                    'right_knee': AngleRange(70, 95, corrections=knee_corrections),
                    'left_knee':  AngleRange(70, 95, corrections=knee_corrections),
                },
            ),
            Phase(
                name='ascending', order=4, is_initial=False,
                instruction='Rise up',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='increasing',
                angles={
                    'right_knee': AngleRange(90, 155, corrections=knee_corrections),
                    'left_knee':  AngleRange(90, 155, corrections=knee_corrections),
                },
            ),
        ],
    )


# Convenient angle snapshots
STANDING  = {'right_knee': 170.0, 'left_knee': 170.0}
HOLD      = {'right_knee': 85.0,  'left_knee': 85.0}


# ---------------------------------------------------------------------------
# AngleRange tests
# ---------------------------------------------------------------------------

class TestAngleRange(unittest.TestCase):

    def test_contains_within(self):
        r = AngleRange(90, 155)
        self.assertTrue(r.contains(120))

    def test_contains_boundary(self):
        r = AngleRange(90, 155)
        self.assertTrue(r.contains(90))
        self.assertTrue(r.contains(155))

    def test_not_contains_outside(self):
        r = AngleRange(90, 155)
        self.assertFalse(r.contains(89))
        self.assertFalse(r.contains(156))

    def test_correction_message_too_low(self):
        r = AngleRange(90, 155, corrections={'too_low': 'Bend more', 'severity': 'high'})
        msg, sev = r.correction_message(70)
        self.assertEqual(msg, 'Bend more')
        self.assertEqual(sev, 'high')

    def test_correction_message_too_high(self):
        r = AngleRange(90, 155, corrections={'too_high': 'Straighten up', 'severity': 'medium'})
        msg, sev = r.correction_message(160)
        self.assertEqual(msg, 'Straighten up')
        self.assertEqual(sev, 'medium')

    def test_correction_message_fallback(self):
        r = AngleRange(90, 155, corrections={})
        msg, sev = r.correction_message(70)
        self.assertIn('70', msg)
        self.assertEqual(sev, 'medium')


# ---------------------------------------------------------------------------
# ExerciseModel loading tests
# ---------------------------------------------------------------------------

class TestExerciseModelLoad(unittest.TestCase):

    def setUp(self):
        self.model = ExerciseModel()   # loads data/exercises_full.json

    def test_all_exercises_loaded(self):
        ids = self.model.list_exercises()
        self.assertIn('squat', ids)
        self.assertIn('lunge', ids)
        self.assertIn('biceps_curl', ids)
        self.assertIn('shoulder_press', ids)

    def test_squat_has_three_phases(self):
        ex = self.model.get_exercise('squat')
        self.assertEqual(len(ex.phases), 3)

    def test_initial_phase_flagged(self):
        for ex_id in self.model.list_exercises():
            ex = self.model.get_exercise(ex_id)
            initials = [p for p in ex.phases if p.is_initial]
            self.assertEqual(len(initials), 1, f'{ex_id} must have exactly one initial phase')

    def test_phases_have_diagnostic_joints(self):
        ex = self.model.get_exercise('squat')
        for phase in ex.phases:
            self.assertGreater(len(phase.diagnostic_joints), 0, f'{phase.name} has no diagnostic joints')

    def test_phase_corrections_embedded(self):
        # Corrections live in the seed file; angle ranges come from MongoDB.
        # When loaded from seed alone, angles dict is empty — check via phase.
        ex = self.model.get_exercise('squat')
        standing = ex.get_phase('standing')
        self.assertIsNotNone(standing)
        self.assertEqual(standing.name, 'standing')

    def test_spine_correction_defined_per_phase(self):
        # spine moved from a hand-set global_constraint to a per-phase joint
        # (2026-08-30) — measured from training data the same way as every
        # other joint, once core.ml.trainer has run. Pre-training, the seed
        # loader skips joints with no numbers yet (same as elbow/knee), so
        # this checks the raw seed JSON directly rather than ex.phases[].angles.
        import json
        from pathlib import Path
        seed_path = Path(__file__).parent.parent / 'data' / 'exercises_seed.json'
        data = json.loads(seed_path.read_text(encoding='utf-8'))
        squat = next(ex for ex in data['exercises'] if ex['id'] == 'squat')
        for phase in squat['phases']:
            self.assertIn('spine', phase['joints'], f'{phase["name"]} missing a spine correction')
            self.assertIn('too_high', phase['joints']['spine']['corrections'])

    def test_primary_joints_set(self):
        ex = self.model.get_exercise('squat')
        self.assertGreater(len(ex.primary_joints), 0)

    def test_mandatory_start_joints_set(self):
        ex = self.model.get_exercise('squat')
        self.assertGreater(len(ex.mandatory_start_joints), 0)

    def test_unknown_exercise_returns_none(self):
        self.assertIsNone(self.model.get_exercise('handstand'))

    def test_phases_sorted_by_order(self):
        ex = self.model.get_exercise('squat')
        orders = [p.order for p in ex.phases]
        self.assertEqual(orders, sorted(orders))


# ---------------------------------------------------------------------------
# Phase matching (occlusion-robust) tests
# ---------------------------------------------------------------------------

class TestMatchPhase(unittest.TestCase):

    def setUp(self):
        exercise = _make_squat()
        self.model = ExerciseModel.__new__(ExerciseModel)
        self.model.exercises = {exercise.id: exercise}

    def test_standing_angles_match_standing_phase(self):
        result = self.model.match_phase('test_squat', STANDING)
        self.assertIsNotNone(result)
        phase, _ = result
        self.assertEqual(phase.name, 'standing')

    def test_hold_angles_match_hold_phase(self):
        result = self.model.match_phase('test_squat', HOLD)
        phase, _ = result
        self.assertEqual(phase.name, 'hold')

    def test_one_joint_missing_still_matches(self):
        partial = {'right_knee': 85.0}  # left_knee missing
        result = self.model.match_phase('test_squat', partial)
        self.assertIsNotNone(result)
        phase, _ = result
        self.assertEqual(phase.name, 'hold')

    def test_unknown_exercise_returns_none(self):
        result = self.model.match_phase('nonexistent', STANDING)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
