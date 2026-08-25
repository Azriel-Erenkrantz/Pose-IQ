"""
Weight tracking + progressive-overload recommendations.

Unit tests for the recommendation rules (pure, no DB) and integration tests
for the API surface: logging a weight on a session and reading back
per-exercise weight recommendations.

Run from the project root:
    python -m pytest tests/test_overload.py -v
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import mongomock

from api.app import app
from core.app_model import (
    FitnessLevel,
    LiveSessionOutput,
    RegisterRequest,
    User,
)
from core.recommendation.overload import (
    FORM_BREAKDOWN_SCORE,
    FORM_READY_SCORE,
    INCREMENT_KG,
    recommend_weight,
    recommend_weights_for_user,
)
from core.user import service
from core.user.workout_history import RepRecord, WorkoutHistory, new_session, rep_form_score

BASE_DATE = datetime(2026, 1, 1)


def _session(exercise_id='biceps_curl', weight=None, score=90.0, day=0):
    """Minimal LiveSessionOutput; `day` orders sessions (higher = newer)."""
    return LiveSessionOutput(
        session_id       = f'{exercise_id}-{day}',
        exercise_id      = exercise_id,
        exercise_name    = exercise_id.replace('_', ' ').title(),
        user_id          = 'u1',
        date             = BASE_DATE + timedelta(days=day),
        reps             = [],
        duration_seconds = 60.0,
        overall_score    = score,
        weight_kg        = weight,
    )


# ---------------------------------------------------------------------------
# Rule unit tests
# ---------------------------------------------------------------------------

class TestRecommendWeight(unittest.TestCase):

    def test_no_sessions_recommends_starting_light(self):
        rec = recommend_weight('biceps_curl', 'Biceps Curl', [])
        self.assertEqual(rec.recommended_weight_kg, 0.0)
        self.assertLess(rec.confidence, 0.5)

    def test_sessions_without_logged_weight_treated_as_no_data(self):
        sessions = [_session(weight=None, score=95, day=d) for d in range(3)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions)
        self.assertEqual(rec.recommended_weight_kg, 0.0)
        self.assertIn('log', rec.reasoning.lower())

    def test_two_clean_sessions_progress_for_intermediate(self):
        sessions = [_session(weight=8.0, score=90, day=d) for d in (1, 2)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions,
                               FitnessLevel.INTERMEDIATE)
        self.assertEqual(rec.recommended_weight_kg, 8.0 + INCREMENT_KG['biceps_curl'])

    def test_beginner_needs_three_clean_sessions(self):
        two = [_session(weight=8.0, score=90, day=d) for d in (1, 2)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', two,
                               FitnessLevel.BEGINNER)
        self.assertEqual(rec.recommended_weight_kg, 8.0)   # hold

        three = two + [_session(weight=8.0, score=90, day=3)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', three,
                               FitnessLevel.BEGINNER)
        self.assertEqual(rec.recommended_weight_kg, 8.0 + INCREMENT_KG['biceps_curl'])

    def test_form_breakdown_backs_off(self):
        sessions = [
            _session(weight=10.0, score=92, day=1),
            _session(weight=10.0, score=FORM_BREAKDOWN_SCORE - 5, day=2),  # newest
        ]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions)
        self.assertEqual(rec.recommended_weight_kg,
                         10.0 - INCREMENT_KG['biceps_curl'])

    def test_back_off_never_goes_below_zero(self):
        sessions = [_session(weight=0.0, score=50, day=1)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions)
        self.assertEqual(rec.recommended_weight_kg, 0.0)

    def test_mediocre_form_holds_weight(self):
        # Above breakdown but below the clean threshold → consolidate
        sessions = [_session(weight=8.0, score=FORM_READY_SCORE - 5, day=d)
                    for d in (1, 2, 3)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions)
        self.assertEqual(rec.recommended_weight_kg, 8.0)

    def test_weight_change_resets_the_streak(self):
        # Two clean sessions at 8 kg, then user moved to 10 kg with one clean
        # session — history at 8 kg says nothing about readiness at 10.
        sessions = [
            _session(weight=8.0, score=95, day=1),
            _session(weight=8.0, score=95, day=2),
            _session(weight=10.0, score=95, day=3),
        ]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions)
        self.assertEqual(rec.recommended_weight_kg, 10.0)   # hold at new weight

    def test_other_exercises_are_ignored(self):
        sessions = [
            _session('squat', weight=40.0, score=95, day=d) for d in (1, 2)
        ] + [_session('biceps_curl', weight=8.0, score=60, day=3)]
        rec = recommend_weight('biceps_curl', 'Biceps Curl', sessions)
        self.assertEqual(rec.recommended_weight_kg,
                         8.0 - INCREMENT_KG['biceps_curl'])

    def test_squat_uses_bigger_increment(self):
        sessions = [_session('squat', weight=40.0, score=95, day=d) for d in (1, 2)]
        rec = recommend_weight('squat', 'Squat', sessions)
        self.assertEqual(rec.recommended_weight_kg, 40.0 + INCREMENT_KG['squat'])


class TestRecommendWeightsForUser(unittest.TestCase):

    def _user(self, level=FitnessLevel.INTERMEDIATE):
        return User(user_id='u1', name='A', email='a@x.com', fitness_level=level)

    def test_one_recommendation_per_trained_exercise(self):
        sessions = [
            _session('squat', weight=40.0, score=95, day=1),
            _session('biceps_curl', weight=8.0, score=95, day=2),
        ]
        recs = recommend_weights_for_user(self._user(), sessions)
        self.assertEqual({r.exercise_id for r in recs}, {'squat', 'biceps_curl'})

    def test_most_recently_trained_first(self):
        sessions = [
            _session('squat', weight=40.0, score=95, day=1),
            _session('biceps_curl', weight=8.0, score=95, day=2),
        ]
        recs = recommend_weights_for_user(self._user(), sessions)
        self.assertEqual(recs[0].exercise_id, 'biceps_curl')

    def test_fitness_level_is_respected(self):
        sessions = [_session(weight=8.0, score=95, day=d) for d in (1, 2)]
        beginner = recommend_weights_for_user(self._user(FitnessLevel.BEGINNER), sessions)
        intermediate = recommend_weights_for_user(self._user(), sessions)
        self.assertEqual(beginner[0].recommended_weight_kg, 8.0)
        self.assertGreater(intermediate[0].recommended_weight_kg, 8.0)

    def test_no_history_returns_empty(self):
        self.assertEqual(recommend_weights_for_user(self._user(), []), [])


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

class MongoMockMixin(unittest.TestCase):
    def setUp(self):
        self._mock_db = mongomock.MongoClient()['poseiq_test']
        self._patch_svc = patch('core.user.service.get_db', return_value=self._mock_db)
        self._patch_wh  = patch('core.user.workout_history.get_db', return_value=self._mock_db)
        # api/app.py's before_request trains the recommendation ranker from
        # Mongo on the first request in this process — patch it too, or the
        # test client's first request reaches out to a real database.
        self._patch_rank = patch('core.recommendation.ratings_service.get_db', return_value=self._mock_db)
        self._patch_scores = patch('core.recommendation.scores_service.get_db', return_value=self._mock_db)
        self._patch_svc.start()
        self._patch_wh.start()
        self._patch_rank.start()
        self._patch_scores.start()
        self.client = app.test_client()

    def tearDown(self):
        self._patch_svc.stop()
        self._patch_wh.stop()
        self._patch_rank.stop()
        self._patch_scores.stop()


def _register(email='athlete@x.com'):
    return service.register(RegisterRequest(
        name='Athlete', email=email, password='secret123',
        fitness_level=FitnessLevel.INTERMEDIATE,
        limitations=[],
    ))


def _save_session(user_id, exercise_id='biceps_curl', weight=None, scores=(90.0, 90.0)):
    session = new_session(exercise_id, exercise_id.title(), weight_kg=weight)
    for i, s in enumerate(scores, start=1):
        session.rep_records.append(RepRecord(rep_number=i, error_joints=[], form_score=s))
    session.total_reps = len(scores)
    WorkoutHistory(user_id=user_id).save_session(session)
    return session


class TestWeightEndpoints(MongoMockMixin):

    def setUp(self):
        super().setUp()
        token = _register()
        self.uid = token.user_id
        self.auth = {'Authorization': f'Bearer {token.token}'}
        self.session = _save_session(self.uid)

    def _put_weight(self, session_id, body, headers=None):
        return self.client.put(
            f'/api/user/{self.uid}/sessions/{session_id}/weight',
            json=body, headers=headers or self.auth)

    def test_set_weight_persists_to_history(self):
        res = self._put_weight(self.session.session_id, {'weight_kg': 8.5})
        self.assertEqual(res.status_code, 200)
        history = self.client.get(f'/api/user/{self.uid}/history',
                                  headers=self.auth).get_json()
        self.assertEqual(history[0]['weight_kg'], 8.5)

    def test_weight_can_be_cleared_with_null(self):
        self._put_weight(self.session.session_id, {'weight_kg': 8.5})
        res = self._put_weight(self.session.session_id, {'weight_kg': None})
        self.assertEqual(res.status_code, 200)
        history = self.client.get(f'/api/user/{self.uid}/history',
                                  headers=self.auth).get_json()
        self.assertIsNone(history[0]['weight_kg'])

    def test_missing_field_rejected(self):
        self.assertEqual(self._put_weight(self.session.session_id, {}).status_code, 400)

    def test_non_numeric_weight_rejected(self):
        res = self._put_weight(self.session.session_id, {'weight_kg': 'heavy'})
        self.assertEqual(res.status_code, 400)

    def test_out_of_range_weight_rejected(self):
        self.assertEqual(self._put_weight(self.session.session_id,
                                          {'weight_kg': -1}).status_code, 400)
        self.assertEqual(self._put_weight(self.session.session_id,
                                          {'weight_kg': 301}).status_code, 400)

    def test_unknown_session_is_404(self):
        self.assertEqual(self._put_weight('no-such-id',
                                          {'weight_kg': 8.0}).status_code, 404)

    def test_cannot_set_weight_on_another_users_session(self):
        other = _register(email='other@x.com')
        other_session = _save_session(other.user_id)
        # Authenticated as self, targeting the other user's session id:
        # scoped lookup must not find it.
        res = self._put_weight(other_session.session_id, {'weight_kg': 8.0})
        self.assertEqual(res.status_code, 404)

    def test_requires_auth(self):
        res = self.client.put(
            f'/api/user/{self.uid}/sessions/{self.session.session_id}/weight',
            json={'weight_kg': 8.0})
        self.assertEqual(res.status_code, 401)


class TestWeightRecommendationEndpoints(MongoMockMixin):

    def setUp(self):
        super().setUp()
        token = _register()
        self.uid = token.user_id
        self.auth = {'Authorization': f'Bearer {token.token}'}

    def test_weights_endpoint_recommends_progression(self):
        # Two clean sessions at 8 kg → intermediate should be told to add load
        _save_session(self.uid, weight=8.0)
        _save_session(self.uid, weight=8.0)
        res = self.client.get(f'/api/user/{self.uid}/weights', headers=self.auth)
        self.assertEqual(res.status_code, 200)
        recs = res.get_json()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['exercise_id'], 'biceps_curl')
        self.assertEqual(recs[0]['recommended_weight_kg'],
                         8.0 + INCREMENT_KG['biceps_curl'])
        self.assertIn('reasoning', recs[0])

    def test_weights_endpoint_empty_without_history(self):
        res = self.client.get(f'/api/user/{self.uid}/weights', headers=self.auth)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), [])

    def test_dashboard_embeds_weight_recommendations(self):
        _save_session(self.uid, weight=8.0)
        res = self.client.get(f'/api/dashboard/{self.uid}', headers=self.auth)
        self.assertEqual(res.status_code, 200)
        dash = res.get_json()
        self.assertIn('weight_recommendations', dash)
        self.assertEqual(len(dash['weight_recommendations']), 1)
        self.assertEqual(dash['weight_recommendations'][0]['exercise_id'],
                         'biceps_curl')


if __name__ == '__main__':
    unittest.main()
