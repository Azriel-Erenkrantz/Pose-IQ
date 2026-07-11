"""
End-to-end integration tests: the full loop from a live workout to the dashboard.

Simulates exactly what core/pipeline.py does at the end of a session
(new_session → RepRecords → WorkoutHistory.save_session) and verifies the
result flows through the API: history, progress metrics, and dashboard
recommendations.

Run from the project root:
    python -m pytest tests/test_api_integration.py -v
"""
import hashlib
import unittest
from unittest.mock import patch

import mongomock

from api.app import app
from core.app_model import (
    Equipment,
    FitnessLevel,
    RegisterRequest,
    TargetGoal,
    TrainerPersonality,
)
from core.user import service
from core.user.workout_history import (
    RepRecord,
    WorkoutHistory,
    new_session,
    rep_form_score,
)


class MongoMockMixin(unittest.TestCase):
    def setUp(self):
        self._mock_db = mongomock.MongoClient()['poseiq_test']
        self._patch_svc = patch('core.user.service.get_db', return_value=self._mock_db)
        self._patch_wh  = patch('core.user.workout_history.get_db', return_value=self._mock_db)
        self._patch_svc.start()
        self._patch_wh.start()
        self.client = app.test_client()

    def tearDown(self):
        self._patch_svc.stop()
        self._patch_wh.stop()


def _register(email='athlete@x.com', goals=(TargetGoal.LEGS,)):
    return service.register(RegisterRequest(
        name='Athlete',
        email=email,
        password='secret123',
        fitness_level=FitnessLevel.INTERMEDIATE,
        trainer_personality=TrainerPersonality.MOTIVATING,
        target_goals=list(goals),
        equipment=[Equipment.NONE],
        limitations=[],
    ))


def _simulate_pipeline_session(user_id: str, exercise_id='squat', exercise_name='Squat',
                               reps_errors=(['spine'], [], [])):
    """Persist a workout session exactly the way core/pipeline.py does."""
    session = new_session(exercise_id, exercise_name)
    for i, errors in enumerate(reps_errors, start=1):
        session.rep_records.append(RepRecord(
            rep_number=i,
            error_joints=list(errors),
            form_score=rep_form_score(list(errors)),
        ))
    session.total_reps = len(reps_errors)
    session.duration_seconds = 42.0
    WorkoutHistory(user_id=user_id).save_session(session)
    return session


class TestWorkoutToDashboardLoop(MongoMockMixin):
    """A session saved by the pipeline must show up in every API surface."""

    def setUp(self):
        super().setUp()
        self.token = _register()
        self.uid = self.token.user_id
        self.auth = {'Authorization': f'Bearer {self.token.token}'}
        self.session = _simulate_pipeline_session(self.uid)

    def test_session_appears_in_history(self):
        res = self.client.get(f'/api/user/{self.uid}/history', headers=self.auth)
        self.assertEqual(res.status_code, 200)
        history = res.get_json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['exercise_id'], 'squat')
        self.assertEqual(len(history[0]['reps']), 3)
        # rep 1 had a spine error → 85, reps 2-3 perfect → 100 ⇒ avg 95
        self.assertEqual(history[0]['overall_score'], 95.0)

    def test_progress_metrics_computed(self):
        res = self.client.get(f'/api/user/{self.uid}/progress', headers=self.auth)
        self.assertEqual(res.status_code, 200)
        progress = res.get_json()
        self.assertEqual(len(progress), 1)
        squat = progress[0]
        self.assertEqual(squat['exercise_id'], 'squat')
        self.assertEqual(squat['total_sessions'], 1)
        self.assertEqual(squat['total_reps'], 3)
        self.assertIn(['spine', 1], squat['weak_joints'])

    def test_dashboard_shows_session_and_recommendations(self):
        res = self.client.get(f'/api/dashboard/{self.uid}', headers=self.auth)
        self.assertEqual(res.status_code, 200)
        dash = res.get_json()
        self.assertEqual(dash['user']['user_id'], self.uid)
        self.assertEqual(len(dash['recent_sessions']), 1)
        self.assertEqual(dash['recent_sessions'][0]['exercise_id'], 'squat')
        # LEGS goal → lower-body recommendations from the catalog
        self.assertGreater(len(dash['recommendations']), 0)
        for rec in dash['recommendations']:
            self.assertIn('score', rec)
            self.assertIn('reason', rec)

    def test_multiple_sessions_accumulate(self):
        _simulate_pipeline_session(self.uid, reps_errors=([], [], [], []))
        res = self.client.get(f'/api/user/{self.uid}/history', headers=self.auth)
        self.assertEqual(len(res.get_json()), 2)
        res = self.client.get(f'/api/user/{self.uid}/progress', headers=self.auth)
        self.assertEqual(res.get_json()[0]['total_reps'], 7)


class TestAuthBoundaries(MongoMockMixin):
    """Regression tests for the IDOR fix: a token only opens its own user's data."""

    def setUp(self):
        super().setUp()
        self.alice = _register(email='alice@x.com')
        self.bob   = _register(email='bob@x.com')

    def test_own_data_allowed(self):
        res = self.client.get(
            f'/api/user/{self.alice.user_id}',
            headers={'Authorization': f'Bearer {self.alice.token}'},
        )
        self.assertEqual(res.status_code, 200)

    def test_other_users_data_forbidden(self):
        for route in (f'/api/user/{self.bob.user_id}',
                      f'/api/user/{self.bob.user_id}/history',
                      f'/api/dashboard/{self.bob.user_id}'):
            res = self.client.get(
                route, headers={'Authorization': f'Bearer {self.alice.token}'})
            self.assertEqual(res.status_code, 403, route)

    def test_other_users_profile_not_writable(self):
        res = self.client.put(
            f'/api/user/{self.bob.user_id}',
            headers={'Authorization': f'Bearer {self.alice.token}'},
            json={'name': 'Hacked'},
        )
        self.assertEqual(res.status_code, 403)

    def test_no_token_unauthorized(self):
        res = self.client.get(f'/api/user/{self.alice.user_id}')
        self.assertEqual(res.status_code, 401)


class TestPasswordHashing(MongoMockMixin):
    """scrypt hashing + transparent upgrade of legacy sha256 accounts."""

    def test_new_accounts_use_scrypt(self):
        token = _register(email='new@x.com')
        doc = self._mock_db.users.find_one({'_id': token.user_id})
        self.assertTrue(doc['password_hash'].startswith('scrypt$'))

    def test_legacy_sha256_login_works_and_upgrades(self):
        token = _register(email='old@x.com')
        # Downgrade the stored hash to the legacy unsalted format
        legacy = hashlib.sha256(b'secret123').hexdigest()
        self._mock_db.users.update_one({'_id': token.user_id},
                                       {'$set': {'password_hash': legacy}})

        res = self.client.post('/api/auth/login',
                               json={'email': 'old@x.com', 'password': 'secret123'})
        self.assertEqual(res.status_code, 200)

        doc = self._mock_db.users.find_one({'_id': token.user_id})
        self.assertTrue(doc['password_hash'].startswith('scrypt$'))

    def test_wrong_password_still_rejected(self):
        _register(email='pw@x.com')
        res = self.client.post('/api/auth/login',
                               json={'email': 'pw@x.com', 'password': 'nope'})
        self.assertEqual(res.status_code, 401)


if __name__ == '__main__':
    unittest.main()
