"""
Rating-based recommendation ranker.

Unit tests for the hand-rolled gradient descent (core/recommendation/
ranker.py) and the ratings storage layer (ratings_service.py).

Run from the project root:
    python -m pytest tests/test_recommendation.py -v
"""
import unittest
from unittest.mock import patch

import mongomock

from core.recommendation import ratings_service
from core.recommendation.catalog import CATALOG
from core.recommendation.ranker import _train_one, recommend_for_user, train


# ── Gradient descent ─────────────────────────────────────────────────────────

class TestTrainOne(unittest.TestCase):

    def test_converges_on_a_known_linear_function(self):
        # y = 2*x0 - 1*x1 + 3, no noise — gradient descent should recover
        # weights and bias close to [2, -1] / 3.
        X = [[1, 1], [2, 1], [1, 3], [4, 2], [3, 3], [2, 4]]
        y = [2 * x0 - 1 * x1 + 3 for x0, x1 in X]

        weights, bias = _train_one(X, y, epochs=3000, lr=0.01)

        self.assertAlmostEqual(weights[0], 2.0, delta=0.2)
        self.assertAlmostEqual(weights[1], -1.0, delta=0.2)
        self.assertAlmostEqual(bias, 3.0, delta=0.5)


# ── Ranker trained on the real fake_ratings.py dataset ────────────────────────

class TestRecommendForUser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        train()  # writes data/recommendation_ranker.json from fake_ratings.py

    def test_cold_start_returns_all_exercises_with_no_ratings_yet_reason(self):
        results = recommend_for_user({}, CATALOG)
        self.assertEqual(len(results), len(CATALOG))
        for r in results:
            self.assertEqual(r.reason_code, 'no_ratings_yet')
            self.assertTrue(0.0 <= r.score <= 1.0)

    def test_rated_exercise_shows_own_rating_not_a_prediction(self):
        results = recommend_for_user({'squat': 5}, CATALOG)
        squat = next(r for r in results if r.exercise.exercise_id == 'squat')
        self.assertEqual(squat.reason_code, 'rated_by_you')
        self.assertEqual(squat.score, 1.0)  # (5-1)/4

    def test_rating_squat_high_ranks_lunge_above_upper_body(self):
        # fake_ratings.py's "lower-body fan" archetype rates squat/lunge
        # high and biceps_curl/shoulder_press low together — a user who
        # loved squat should get lunge predicted above the upper-body pair.
        results = recommend_for_user({'squat': 5}, CATALOG)
        by_id = {r.exercise.exercise_id: r.score for r in results}
        self.assertGreater(by_id['lunge'], by_id['biceps_curl'])
        self.assertGreater(by_id['lunge'], by_id['shoulder_press'])

    def test_unrated_exercises_are_predicted_from_ratings(self):
        results = recommend_for_user({'squat': 5}, CATALOG)
        for r in results:
            if r.exercise.exercise_id != 'squat':
                self.assertEqual(r.reason_code, 'predicted_from_ratings')


# ── Ratings storage ────────────────────────────────────────────────────────────

class TestRatingsService(unittest.TestCase):

    def setUp(self):
        self._mock_db = mongomock.MongoClient()['poseiq_test']
        self._patch = patch('core.recommendation.ratings_service.get_db', return_value=self._mock_db)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_save_then_get_round_trips(self):
        ratings_service.save_rating('u1', 'squat', 4)
        ratings_service.save_rating('u1', 'lunge', 2)
        self.assertEqual(ratings_service.get_user_ratings('u1'), {'squat': 4, 'lunge': 2})

    def test_re_rating_overwrites_not_duplicates(self):
        ratings_service.save_rating('u1', 'squat', 3)
        ratings_service.save_rating('u1', 'squat', 5)
        self.assertEqual(ratings_service.get_user_ratings('u1'), {'squat': 5})

    def test_ratings_are_scoped_per_user(self):
        ratings_service.save_rating('u1', 'squat', 4)
        ratings_service.save_rating('u2', 'squat', 1)
        self.assertEqual(ratings_service.get_user_ratings('u1'), {'squat': 4})
        self.assertEqual(ratings_service.get_user_ratings('u2'), {'squat': 1})


if __name__ == '__main__':
    unittest.main()
