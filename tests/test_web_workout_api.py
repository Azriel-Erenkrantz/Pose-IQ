"""
API surface for the web live-workout screen:
  GET  /api/exercises            — full exercise model (phases + angle ranges)
  POST /api/user/<id>/sessions   — save a workout completed in the browser

Run from the project root:
    python -m pytest tests/test_web_workout_api.py -v
"""
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


class MongoMockMixin(unittest.TestCase):
    def setUp(self):
        self._mock_db = mongomock.MongoClient()['poseiq_test']
        self._patches = [
            patch('core.user.service.get_db', return_value=self._mock_db),
            patch('core.user.workout_history.get_db', return_value=self._mock_db),
            patch('core.db.get_db', return_value=self._mock_db),
            # api/app.py's before_request trains the recommendation ranker
            # from Mongo on the first request in this process.
            patch('core.recommendation.ratings_service.get_db', return_value=self._mock_db),
        ]
        for p in self._patches:
            p.start()
        self.client = app.test_client()

    def tearDown(self):
        for p in self._patches:
            p.stop()


def _register(email='athlete@x.com'):
    return service.register(RegisterRequest(
        name='Athlete', email=email, password='secret123',
        fitness_level=FitnessLevel.INTERMEDIATE,
        trainer_personality=TrainerPersonality.MOTIVATING,
        target_goals=[TargetGoal.LEGS], equipment=[Equipment.NONE],
        limitations=[],
    ))


def _seed_squat(db, with_angles=True):
    """Minimal squat doc in the exercises collection (+ measured ranges)."""
    db.exercises.insert_one({
        'id': 'squat', 'name': 'Squat', 'description': 'Basic squat',
        'muscle_groups': ['quads'],
        'mandatory_start_joints': ['right_knee', 'left_knee'],
        'global_constraints': {
            'spine': {'min': 0, 'max': 30,
                      'corrections': {'too_high': 'Keep back straight', 'severity': 'high'}},
        },
        'phases': [
            {'name': 'standing', 'order': 1, 'is_initial': True,
             'instruction': 'Stand straight',
             'diagnostic_joints': ['right_knee', 'left_knee'],
             'motion_direction': 'stable',
             'joints': {'right_knee': {'corrections': {'too_low': 'Straighten up'}},
                        'left_knee':  {'corrections': {}}}},
            {'name': 'descending', 'order': 2,
             'instruction': 'Squat down',
             'diagnostic_joints': ['right_knee', 'left_knee'],
             'motion_direction': 'decreasing',
             'joints': {'right_knee': {'corrections': {}},
                        'left_knee':  {'corrections': {}}}},
            {'name': 'ascending', 'order': 3,
             'instruction': 'Rise up',
             'diagnostic_joints': ['right_knee', 'left_knee'],
             'motion_direction': 'increasing',
             'joints': {'right_knee': {'corrections': {}},
                        'left_knee':  {'corrections': {}}}},
        ],
    })
    if with_angles:
        for phase, lo, hi in (('standing', 155, 180), ('descending', 90, 155),
                              ('ascending', 90, 160)):
            db.exercise_angles.insert_one({
                'exercise_id': 'squat', 'phase': phase,
                'joints': {
                    'right_knee': {'min': lo, 'max': hi, 'mean': (lo + hi) / 2, 'std': 8.0},
                    'left_knee':  {'min': lo, 'max': hi, 'mean': (lo + hi) / 2, 'std': 8.0},
                },
            })


class TestExercisesEndpoint(MongoMockMixin):

    def test_returns_full_model_with_ranges(self):
        _seed_squat(self._mock_db)
        res = self.client.get('/api/exercises')
        self.assertEqual(res.status_code, 200)
        exercises = res.get_json()
        squat = next(e for e in exercises if e['id'] == 'squat')

        self.assertTrue(squat['ready'])
        self.assertEqual(len(squat['phases']), 3)
        standing = squat['phases'][0]
        self.assertEqual(standing['name'], 'standing')
        self.assertTrue(standing['is_initial'])
        self.assertEqual(standing['angles']['right_knee']['min'], 155)
        self.assertEqual(standing['angles']['right_knee']['max'], 180)
        self.assertEqual(standing['angles']['right_knee']['corrections']['too_low'],
                         'Straighten up')
        self.assertEqual(squat['global_constraints']['spine']['max'], 30)
        self.assertIn('right_knee', squat['mandatory_start_joints'])

    def test_not_ready_without_trained_ranges(self):
        _seed_squat(self._mock_db, with_angles=False)
        res = self.client.get('/api/exercises')
        squat = next(e for e in res.get_json() if e['id'] == 'squat')
        self.assertFalse(squat['ready'])
        self.assertEqual(squat['phases'][0]['angles'], {})

    def test_no_auth_required(self):
        _seed_squat(self._mock_db)
        res = self.client.get('/api/exercises')
        self.assertEqual(res.status_code, 200)

    def test_empty_db_falls_back_to_seed_json(self):
        # No exercises in Mongo → the real seed file is served (not ready)
        res = self.client.get('/api/exercises')
        self.assertEqual(res.status_code, 200)
        ids = {e['id'] for e in res.get_json()}
        self.assertIn('squat', ids)
        self.assertTrue(all(e['ready'] is False for e in res.get_json()))


VALID_BODY = {
    'exercise_id': 'squat',
    'duration_seconds': 95.5,
    'weight_kg': 20.0,
    'reps': [
        {'rep_number': 1, 'error_joints': [], 'form_score': 100.0},
        {'rep_number': 2, 'error_joints': ['spine'], 'form_score': 85.0},
        {'rep_number': 3, 'error_joints': [], 'form_score': 100.0},
    ],
}


class TestCreateSessionEndpoint(MongoMockMixin):

    def setUp(self):
        super().setUp()
        _seed_squat(self._mock_db)
        token = _register()
        self.uid = token.user_id
        self.auth = {'Authorization': f'Bearer {token.token}'}

    def _post(self, body, headers=None):
        return self.client.post(f'/api/user/{self.uid}/sessions',
                                json=body, headers=headers or self.auth)

    def test_session_saved_and_scored(self):
        res = self._post(VALID_BODY)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertEqual(data['total_reps'], 3)
        self.assertEqual(data['overall_score'], 95.0)   # (100+85+100)/3
        self.assertEqual(data['weight_kg'], 20.0)

    def test_session_appears_in_history(self):
        self._post(VALID_BODY)
        history = self.client.get(f'/api/user/{self.uid}/history',
                                  headers=self.auth).get_json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['exercise_id'], 'squat')
        self.assertEqual(history[0]['exercise_name'], 'Squat')   # looked up in Mongo
        self.assertEqual(history[0]['weight_kg'], 20.0)
        self.assertEqual(len(history[0]['reps']), 3)
        self.assertEqual(history[0]['reps'][1]['error_joints'], ['spine'])

    def test_feeds_weight_recommendations(self):
        self._post(VALID_BODY)
        self._post(VALID_BODY)
        recs = self.client.get(f'/api/user/{self.uid}/weights',
                               headers=self.auth).get_json()
        self.assertEqual(recs[0]['exercise_id'], 'squat')
        self.assertGreater(recs[0]['recommended_weight_kg'], 20.0)

    def test_missing_exercise_id_rejected(self):
        body = {**VALID_BODY}
        del body['exercise_id']
        self.assertEqual(self._post(body).status_code, 400)

    def test_empty_reps_rejected(self):
        self.assertEqual(self._post({**VALID_BODY, 'reps': []}).status_code, 400)

    def test_malformed_rep_rejected(self):
        bad = {**VALID_BODY, 'reps': [{'error_joints': []}]}   # no rep_number/form_score
        self.assertEqual(self._post(bad).status_code, 400)

    def test_invalid_weight_rejected(self):
        self.assertEqual(self._post({**VALID_BODY, 'weight_kg': 999}).status_code, 400)

    def test_weight_optional(self):
        body = {**VALID_BODY}
        del body['weight_kg']
        res = self._post(body)
        self.assertEqual(res.status_code, 201)
        self.assertIsNone(res.get_json()['weight_kg'])

    def test_requires_auth(self):
        res = self.client.post(f'/api/user/{self.uid}/sessions', json=VALID_BODY)
        self.assertEqual(res.status_code, 401)

    def test_other_users_token_forbidden(self):
        other = _register(email='other@x.com')
        res = self._post(VALID_BODY,
                         headers={'Authorization': f'Bearer {other.token}'})
        self.assertEqual(res.status_code, 403)


if __name__ == '__main__':
    unittest.main()
