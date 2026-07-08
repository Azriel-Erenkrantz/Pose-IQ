"""
Tests for Model 1: exercise loading, phase detection, violation messages, rep counting.

All tests use synthetic angle data — no camera or MediaPipe required.

Run from project root:
    python -m pytest core/exercise/tests.py -v
"""
import unittest
from typing import Dict

from core.exercise.exercise_model import AngleRange, Exercise, ExerciseModel, Phase
from core.exercise.exercise_state_machine import ExerciseStateMachine
from core.exercise.posture_rules import PostureRules


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


def _make_model_and_machine() -> tuple[ExerciseModel, ExerciseStateMachine]:
    exercise = _make_squat()
    model = ExerciseModel.__new__(ExerciseModel)
    model.exercises = {exercise.id: exercise}
    machine = ExerciseStateMachine(exercise, model)
    return model, machine


# Convenient angle snapshots
STANDING  = {'right_knee': 170.0, 'left_knee': 170.0}
DESCEND   = {'right_knee': 120.0, 'left_knee': 120.0}
HOLD      = {'right_knee': 85.0,  'left_knee': 85.0}
ASCEND    = {'right_knee': 120.0, 'left_knee': 120.0}


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
        self.assertIn('plank', ids)

    def test_squat_has_four_phases(self):
        ex = self.model.get_exercise('squat')
        self.assertEqual(len(ex.phases), 4)

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
        ex = self.model.get_exercise('squat')
        standing = ex.get_phase('standing')
        r = standing.angles['right_knee']
        self.assertIn('too_low', r.corrections)
        self.assertIn('severity', r.corrections)

    def test_global_constraints_have_corrections(self):
        ex = self.model.get_exercise('squat')
        self.assertIn('spine', ex.global_constraints)
        spine = ex.global_constraints['spine']
        self.assertIn('too_high', spine.corrections)

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


# ---------------------------------------------------------------------------
# ExerciseStateMachine: readiness and start
# ---------------------------------------------------------------------------

class TestStateMachineReadiness(unittest.TestCase):

    def setUp(self):
        _, self.machine = _make_model_and_machine()

    def test_not_started_when_angles_wrong(self):
        # Correct angles start immediately — wrong angles must NOT start
        result = self.machine.update(HOLD)   # knees bent → not in standing position
        self.assertFalse(result['started'])

    def test_starts_on_first_correct_standing_frame(self):
        result = self.machine.update(STANDING)
        self.assertTrue(result['started'])

    def test_mandatory_joint_missing_blocks_start(self):
        result = self.machine.update({'right_knee': 170.0})  # left_knee missing
        self.assertFalse(result['started'])
        self.assertIn('left_knee', result['readiness'])

    def test_wrong_angles_in_standing_blocks_start(self):
        result = self.machine.update(HOLD)  # knee too bent for standing
        self.assertFalse(result['started'])

    def test_no_angles_does_not_start(self):
        result = self.machine.update({})
        self.assertFalse(result['started'])


# ---------------------------------------------------------------------------
# ExerciseStateMachine: phase transitions and rep counting
# ---------------------------------------------------------------------------

class TestStateMachineTransitions(unittest.TestCase):

    def _start(self, machine):
        for _ in range(ExerciseStateMachine.FRAMES_TO_TRANSITION + 1):
            machine.update(STANDING)

    def _hold_phase(self, machine, angles, frames=None):
        frames = frames or ExerciseStateMachine.FRAMES_TO_TRANSITION + 1
        result = None
        for _ in range(frames):
            result = machine.update(angles)
        return result

    def test_full_rep_increments_rep_count(self):
        _, machine = _make_model_and_machine()
        self._start(machine)
        self._hold_phase(machine, DESCEND)
        self._hold_phase(machine, HOLD)
        self._hold_phase(machine, ASCEND)
        result = self._hold_phase(machine, STANDING)
        self.assertEqual(result['rep_count'], 1)

    def test_two_reps(self):
        _, machine = _make_model_and_machine()
        self._start(machine)
        for _ in range(2):
            self._hold_phase(machine, DESCEND)
            self._hold_phase(machine, HOLD)
            self._hold_phase(machine, ASCEND)
            self._hold_phase(machine, STANDING)
        self.assertEqual(machine.rep_count, 2)

    def test_phase_name_advances(self):
        _, machine = _make_model_and_machine()
        self._start(machine)
        self.assertEqual(machine.current_phase.name, 'standing')
        self._hold_phase(machine, DESCEND)
        self.assertEqual(machine.current_phase.name, 'descending')

    def test_transition_requires_minimum_frames(self):
        _, machine = _make_model_and_machine()
        self._start(machine)
        # Only 2 frames of descending (below the transition threshold)
        machine.update(DESCEND)
        machine.update(DESCEND)
        self.assertEqual(machine.current_phase.name, 'standing')

    def test_instruction_updates_with_phase(self):
        _, machine = _make_model_and_machine()
        self._start(machine)
        self._hold_phase(machine, DESCEND)
        result = machine.update(DESCEND)
        self.assertEqual(result['instruction'], 'Squat down')


# ---------------------------------------------------------------------------
# PostureRules: violation detection and messages
# ---------------------------------------------------------------------------

class TestPostureRules(unittest.TestCase):

    def setUp(self):
        self.exercise = _make_squat()
        self.rules = PostureRules(self.exercise)
        # Phase-only rules (no global constraints) so tests aren't disrupted by missing spine
        hold = self.exercise.get_phase('hold')
        self.active_rules = hold.angles.copy()
        # Separate rules dict that includes global constraints, for the spine test
        self.active_rules_global = {**self.exercise.global_constraints, **hold.angles}

    def _trigger(self, angles, frames=None):
        frames = frames or PostureRules.FRAMES_TO_ALERT + 1
        issues = []
        for _ in range(frames):
            issues = self.rules.analyze(angles, self.active_rules)
        return issues

    def test_correct_form_no_violations(self):
        issues = self._trigger(HOLD)
        self.assertEqual(issues, [])

    def test_too_low_triggers_message(self):
        bad = {'right_knee': 60.0, 'left_knee': 60.0}
        issues = self._trigger(bad)
        joints = [i['joint'] for i in issues]
        self.assertIn('right_knee', joints)
        msg = next(i['message'] for i in issues if i['joint'] == 'right_knee')
        self.assertEqual(msg, 'Bend more')

    def test_too_high_triggers_message(self):
        bad = {'right_knee': 150.0, 'left_knee': 150.0}
        issues = self._trigger(bad)
        joints = [i['joint'] for i in issues]
        self.assertIn('right_knee', joints)
        msg = next(i['message'] for i in issues if i['joint'] == 'right_knee')
        self.assertEqual(msg, 'Straighten up')

    def test_global_constraint_violation(self):
        bad = {**HOLD, 'spine': 45.0}  # spine > 30°
        rules = PostureRules(self.exercise)
        issues = []
        for _ in range(PostureRules.FRAMES_TO_ALERT + 1):
            issues = rules.analyze(bad, self.active_rules_global)
        joints = [i['joint'] for i in issues]
        self.assertIn('spine', joints)
        msg = next(i['message'] for i in issues if i['joint'] == 'spine')
        self.assertEqual(msg, 'Keep back straight')

    def test_violation_below_frame_threshold_not_reported(self):
        bad = {'right_knee': 60.0, 'left_knee': 85.0}
        for _ in range(PostureRules.FRAMES_TO_ALERT - 1):
            issues = self.rules.analyze(bad, self.active_rules)
        self.assertEqual(issues, [])

    def test_missing_joint_reported_after_threshold(self):
        no_knee = {'left_knee': 85.0}   # right_knee missing
        issues = self._trigger(no_knee, frames=PostureRules.FRAMES_TO_MISSING + 1)
        joints = [i['joint'] for i in issues]
        self.assertIn('right_knee', joints)
        missing = next(i for i in issues if i['joint'] == 'right_knee')
        self.assertEqual(missing['direction'], 'missing')

    def test_recovery_clears_counter(self):
        bad = {'right_knee': 60.0, 'left_knee': 60.0}
        self._trigger(bad)
        # Now correct form
        good_issues = self.rules.analyze(HOLD, self.active_rules)
        self.assertEqual(good_issues, [])
        # Wrong again — counter should have reset so no immediate alert
        issues = self.rules.analyze(bad, self.active_rules)
        self.assertEqual(issues, [])

    def test_limited_joint_severity_downgraded(self):
        rules = PostureRules(self.exercise, limited_joints=['right_knee'])
        bad = {'right_knee': 60.0, 'left_knee': 85.0}
        for _ in range(PostureRules.FRAMES_TO_ALERT + 1):
            issues = rules.analyze(bad, self.active_rules)
        knee_issue = next((i for i in issues if i['joint'] == 'right_knee'), None)
        self.assertIsNotNone(knee_issue)
        self.assertEqual(knee_issue['severity'], 'low')
        self.assertIn('[Adapted]', knee_issue['message'])

    def test_reset_clears_all_counters(self):
        bad = {'right_knee': 60.0, 'left_knee': 60.0}
        self._trigger(bad)
        self.rules.reset()
        # After reset, a single bad frame should not trigger alerts
        issues = self.rules.analyze(bad, self.active_rules)
        self.assertEqual(issues, [])


# ---------------------------------------------------------------------------
# Auto-recovery after loss of tracking
# ---------------------------------------------------------------------------

class TestAutoRecovery(unittest.TestCase):

    def test_returns_to_correct_phase_after_disconnect(self):
        _, machine = _make_model_and_machine()
        # Start the machine
        for _ in range(ExerciseStateMachine.FRAMES_TO_TRANSITION + 1):
            machine.update(STANDING)
        # Transition to hold
        for _ in range(ExerciseStateMachine.FRAMES_TO_TRANSITION + 1):
            machine.update(DESCEND)
        for _ in range(ExerciseStateMachine.FRAMES_TO_TRANSITION + 1):
            machine.update(HOLD)

        # Simulate disconnect
        for _ in range(ExerciseStateMachine.FRAMES_TO_DISCONNECT + 1):
            machine.update({})

        # Reappear in hold position — auto-recovery should snap to hold phase
        machine.update(HOLD)
        self.assertEqual(machine.current_phase.name, 'hold')


if __name__ == '__main__':
    unittest.main()
