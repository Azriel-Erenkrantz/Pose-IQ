"""
Tests for core/user/ — service layer, fake data, and progress computation.

Run from the project root:
    python -m pytest core/user/tests.py -v
    # or
    python -m unittest core.user.tests
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from core.app_model import (
    BodyRegion,
    Equipment,
    FitnessLevel,
    LiveSessionOutput,
    LoginRequest,
    RegisterRequest,
    RepResult,
    TargetGoal,
    TrainerPersonality,
    User,
)
from core.user import fake_data, service


# ── Helpers ────────────────────────────────────────────────────────────────────

def _req(email='test@x.com', password='secret123', **kwargs) -> RegisterRequest:
    defaults = dict(
        name='Test User',
        email=email,
        password=password,
        fitness_level=FitnessLevel.INTERMEDIATE,
        trainer_personality=TrainerPersonality.CALM,
        target_goals=[TargetGoal.ABS],
        equipment=[Equipment.NONE],
        limitations=[],
    )
    defaults.update(kwargs)
    return RegisterRequest(**defaults)


def _session(exercise_id='squat', exercise_name='Squat',
             score=80.0, days_ago=1, user_id='u1') -> LiveSessionOutput:
    reps = [RepResult(rep_number=1, form_score=score, error_joints=[], duration_seconds=3.0)]
    return LiveSessionOutput(
        session_id=f's-{exercise_id}-{days_ago}',
        exercise_id=exercise_id,
        exercise_name=exercise_name,
        user_id=user_id,
        date=datetime.now() - timedelta(days=days_ago),
        reps=reps,
        duration_seconds=30.0,
        overall_score=score,
    )


# ── Mixin: patch _USERS_PATH to a temp file ────────────────────────────────────

class TempStoreMixin:
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = patch.object(
            service, '_USERS_PATH',
            os.path.join(self.tmpdir, 'users.json'),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmpdir)


# ── Auth tests ─────────────────────────────────────────────────────────────────

class TestRegister(TempStoreMixin, unittest.TestCase):

    def test_returns_auth_token(self):
        token = service.register(_req())
        self.assertIsNotNone(token)
        self.assertTrue(token.user_id)
        self.assertTrue(token.token)
        self.assertIsInstance(token.expires_at, datetime)

    def test_duplicate_email_returns_none(self):
        service.register(_req(email='dup@x.com'))
        second = service.register(_req(email='dup@x.com'))
        self.assertIsNone(second)

    def test_different_emails_both_succeed(self):
        t1 = service.register(_req(email='a@x.com'))
        t2 = service.register(_req(email='b@x.com'))
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertNotEqual(t1.user_id, t2.user_id)

    def test_profile_fields_are_persisted(self):
        token = service.register(_req(
            email='alice@x.com',
            fitness_level=FitnessLevel.ADVANCED,
            target_goals=[TargetGoal.LEGS, TargetGoal.ABS],
            equipment=[Equipment.DUMBBELLS],
        ))
        user = service.get_user(token.user_id)
        self.assertEqual(user.fitness_level, FitnessLevel.ADVANCED)
        self.assertIn(TargetGoal.LEGS, user.target_goals)
        self.assertIn(Equipment.DUMBBELLS, user.equipment)


class TestLogin(TempStoreMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.token = service.register(_req(email='login@x.com', password='correct'))

    def test_correct_credentials_returns_token(self):
        token = service.login(LoginRequest(email='login@x.com', password='correct'))
        self.assertIsNotNone(token)
        self.assertEqual(token.user_id, self.token.user_id)

    def test_wrong_password_returns_none(self):
        result = service.login(LoginRequest(email='login@x.com', password='wrong'))
        self.assertIsNone(result)

    def test_unknown_email_returns_none(self):
        result = service.login(LoginRequest(email='nobody@x.com', password='pw'))
        self.assertIsNone(result)

    def test_each_login_issues_new_token(self):
        t1 = service.login(LoginRequest(email='login@x.com', password='correct'))
        t2 = service.login(LoginRequest(email='login@x.com', password='correct'))
        self.assertNotEqual(t1.token, t2.token)

    def test_password_is_not_stored_in_plaintext(self):
        # Read the raw JSON file and confirm the password is not visible
        import json
        with open(service._USERS_PATH) as f:
            raw = json.load(f)
        for user_data in raw.values():
            self.assertNotIn('password', user_data)
            self.assertIn('password_hash', user_data)
            self.assertNotEqual(user_data['password_hash'], 'correct')


# ── User CRUD tests ────────────────────────────────────────────────────────────

class TestGetUser(TempStoreMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.token = service.register(_req(name='Alice', email='alice@x.com'))

    def test_returns_user_with_correct_fields(self):
        user = service.get_user(self.token.user_id)
        self.assertIsNotNone(user)
        self.assertEqual(user.name, 'Alice')
        self.assertEqual(user.email, 'alice@x.com')

    def test_unknown_id_returns_none(self):
        self.assertIsNone(service.get_user('does-not-exist'))

    def test_returned_type_is_user(self):
        user = service.get_user(self.token.user_id)
        self.assertIsInstance(user, User)


class TestUpdateUser(TempStoreMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.token = service.register(_req(email='update@x.com'))
        self.user  = service.get_user(self.token.user_id)

    def test_name_change_persists(self):
        self.user.name = 'New Name'
        service.update_user(self.user)
        self.assertEqual(service.get_user(self.token.user_id).name, 'New Name')

    def test_fitness_level_change_persists(self):
        self.user.fitness_level = FitnessLevel.ADVANCED
        service.update_user(self.user)
        self.assertEqual(service.get_user(self.token.user_id).fitness_level, FitnessLevel.ADVANCED)

    def test_update_preserves_password_hash(self):
        import json
        hash_before = json.load(open(service._USERS_PATH))[self.token.user_id]['password_hash']
        self.user.name = 'Changed'
        service.update_user(self.user)
        hash_after = json.load(open(service._USERS_PATH))[self.token.user_id]['password_hash']
        self.assertEqual(hash_before, hash_after)

    def test_unknown_user_returns_false(self):
        ghost = User(user_id='ghost', name='X', email='x@x.com',
                     fitness_level=FitnessLevel.BEGINNER,
                     trainer_personality=TrainerPersonality.CALM)
        self.assertFalse(service.update_user(ghost))


# ── Health status tests ────────────────────────────────────────────────────────

class TestHealthStatus(unittest.TestCase):

    def test_covers_all_body_regions(self):
        status = service.get_health_status('any-id')
        for region in BodyRegion:
            self.assertIn(region, status.ratings)

    def test_default_is_fully_healthy(self):
        status = service.get_health_status('any-id')
        self.assertTrue(all(v == 5 for v in status.ratings.values()))

    def test_get_returns_correct_rating(self):
        status = service.get_health_status('any-id')
        self.assertEqual(status.get(BodyRegion.UPPER), 5)


# ── Profile setup options tests ────────────────────────────────────────────────

class TestProfileSetupOptions(unittest.TestCase):

    def setUp(self):
        self.opts = service.get_profile_setup_options()

    def test_all_fitness_levels_present(self):
        self.assertEqual(set(self.opts.fitness_levels), set(FitnessLevel))

    def test_all_target_goals_present(self):
        self.assertEqual(set(self.opts.target_goals), set(TargetGoal))

    def test_all_equipment_options_present(self):
        self.assertEqual(set(self.opts.equipment_options), set(Equipment))

    def test_all_trainer_personalities_present(self):
        self.assertEqual(set(self.opts.trainer_personalities), set(TrainerPersonality))

    def test_limitation_options_nonempty_strings(self):
        self.assertGreater(len(self.opts.limitation_options), 0)
        self.assertTrue(all(isinstance(l, str) for l in self.opts.limitation_options))


# ── Fake data tests ────────────────────────────────────────────────────────────

class TestFakeUser(unittest.TestCase):

    def test_is_user_instance(self):
        self.assertIsInstance(fake_data.fake_user(), User)

    def test_has_required_fields(self):
        u = fake_data.fake_user()
        self.assertTrue(u.user_id)
        self.assertTrue(u.name)
        self.assertTrue(u.email)

    def test_fitness_level_is_valid_enum(self):
        self.assertIsInstance(fake_data.fake_user().fitness_level, FitnessLevel)

    def test_auth_token_user_id_matches_user(self):
        self.assertEqual(
            fake_data.fake_user().user_id,
            fake_data.fake_auth_token().user_id,
        )


class TestFakeSessions(unittest.TestCase):

    def test_returns_eight_sessions(self):
        self.assertEqual(len(fake_data.fake_sessions()), 8)

    def test_all_scores_in_range(self):
        for s in fake_data.fake_sessions():
            self.assertGreaterEqual(s.overall_score, 0)
            self.assertLessEqual(s.overall_score, 100)

    def test_all_belong_to_fake_user(self):
        for s in fake_data.fake_sessions():
            self.assertEqual(s.user_id, fake_data.FAKE_USER_ID)

    def test_covers_multiple_exercises(self):
        ids = {s.exercise_id for s in fake_data.fake_sessions()}
        self.assertGreater(len(ids), 1)

    def test_each_session_has_reps(self):
        for s in fake_data.fake_sessions():
            self.assertGreater(s.total_reps, 0)


class TestFakeProgress(unittest.TestCase):

    def test_metrics_reference_known_exercises(self):
        session_ids = {s.exercise_id for s in fake_data.fake_sessions()}
        for m in fake_data.fake_progress_metrics():
            self.assertIn(m.exercise_id, session_ids)

    def test_avg_score_in_range(self):
        for m in fake_data.fake_progress_metrics():
            self.assertGreaterEqual(m.avg_score_recent, 0)
            self.assertLessEqual(m.avg_score_recent, 100)

    def test_weak_joints_are_tuples(self):
        for m in fake_data.fake_progress_metrics():
            for item in m.weak_joints:
                self.assertEqual(len(item), 2)
                self.assertIsInstance(item[0], str)
                self.assertIsInstance(item[1], int)


class TestFakeInjuryRisk(unittest.TestCase):

    def test_risk_in_range(self):
        risk = fake_data.fake_injury_risk()
        self.assertGreaterEqual(risk.overall_risk, 0)
        self.assertLessEqual(risk.overall_risk, 1)

    def test_has_recommendation(self):
        self.assertTrue(fake_data.fake_injury_risk().recommendation)

    def test_has_warning_property(self):
        risk = fake_data.fake_injury_risk()
        expected = risk.overall_risk >= 0.4
        self.assertEqual(risk.has_warning, expected)


# ── Progress computation tests ─────────────────────────────────────────────────

class TestProgressComputation(TempStoreMixin, unittest.TestCase):

    def test_empty_history_returns_empty_list(self):
        token = service.register(_req(email='empty@x.com'))
        with patch.object(service, 'get_history', return_value=[]):
            self.assertEqual(service.get_progress(token.user_id), [])

    def test_single_exercise_produces_one_metric(self):
        sessions = [_session('squat', 'Squat', score=75.0, days_ago=i) for i in range(3)]
        with patch.object(service, 'get_history', return_value=sessions):
            metrics = service.get_progress('u1')
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].exercise_id, 'squat')

    def test_two_exercises_produce_two_metrics(self):
        sessions = [
            _session('squat', 'Squat', score=75.0, days_ago=1),
            _session('plank', 'Plank', score=85.0, days_ago=2),
        ]
        with patch.object(service, 'get_history', return_value=sessions):
            metrics = service.get_progress('u1')
        ids = {m.exercise_id for m in metrics}
        self.assertEqual(ids, {'squat', 'plank'})

    def test_improving_trend_detected(self):
        sessions = [
            _session(score=60.0, days_ago=10),
            _session(score=72.0, days_ago=5),
            _session(score=85.0, days_ago=1),
        ]
        with patch.object(service, 'get_history', return_value=sessions):
            metrics = service.get_progress('u1')
        self.assertEqual(metrics[0].score_trend.value, 'improving')

    def test_declining_trend_detected(self):
        sessions = [
            _session(score=85.0, days_ago=10),
            _session(score=72.0, days_ago=5),
            _session(score=60.0, days_ago=1),
        ]
        with patch.object(service, 'get_history', return_value=sessions):
            metrics = service.get_progress('u1')
        self.assertEqual(metrics[0].score_trend.value, 'declining')

    def test_stable_trend_when_scores_close(self):
        sessions = [
            _session(score=75.0, days_ago=3),
            _session(score=77.0, days_ago=1),
        ]
        with patch.object(service, 'get_history', return_value=sessions):
            metrics = service.get_progress('u1')
        self.assertEqual(metrics[0].score_trend.value, 'stable')

    def test_total_reps_counted_correctly(self):
        # Each _session() creates 1 rep
        sessions = [_session(days_ago=i) for i in range(4)]
        with patch.object(service, 'get_history', return_value=sessions):
            metrics = service.get_progress('u1')
        self.assertEqual(metrics[0].total_reps, 4)
        self.assertEqual(metrics[0].total_sessions, 4)


if __name__ == '__main__':
    unittest.main()
