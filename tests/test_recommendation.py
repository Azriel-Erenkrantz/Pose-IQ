"""
Rating-based recommendation ranker.

Unit tests for the hand-rolled gradient descent (core/recommendation/
ranker.py) and the ratings storage layer (ratings_service.py), including
that new ratings stored in Mongo actually influence retraining, not just
each rater's own inference-time input.

Run from the project root:
    python -m pytest tests/test_recommendation.py -v
"""
import random
import unittest
from unittest.mock import patch

import mongomock

from core.recommendation import ratings_service, scores_service
from core.recommendation.catalog import CATALOG
from core.recommendation.ranker import _fit, recommend_for_user, train


# ── Gradient descent ─────────────────────────────────────────────────────────

class TestTrainOne(unittest.TestCase):

    def test_converges_on_a_known_linear_function(self):
        # y = 2*x0 - 1*x1 + 3, no noise — gradient descent should recover
        # weights and bias close to [2, -1] / 3.
        X = [[1, 1], [2, 1], [1, 3], [4, 2], [3, 3], [2, 4]]
        y = [2 * x0 - 1 * x1 + 3 for x0, x1 in X]

        weights, bias = _fit(X, y, epochs=3000, lr=0.01)

        self.assertAlmostEqual(weights[0], 2.0, delta=0.2)
        self.assertAlmostEqual(weights[1], -1.0, delta=0.2)
        self.assertAlmostEqual(bias, 3.0, delta=0.5)


# ── Ranker trained against Mongo ────────────────────────────────────────────

class MongoMockMixin:
    def setUp(self):
        self._mock_db = mongomock.MongoClient()['poseiq_test']
        self._patches = [
            patch('core.recommendation.ratings_service.get_db', return_value=self._mock_db),
            patch('core.recommendation.scores_service.get_db', return_value=self._mock_db),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()


def _seed_correlated_population(rng: random.Random, n_per_group: int = 10) -> None:
    """
    Write a training population into 'ratings' with two groups that pull
    opposite directions — one group rates squat/lunge high and
    biceps_curl/shoulder_press low, the other the reverse — so tests can
    check the model actually learns a correlation, not just run without
    crashing. Ordinary save_rating() calls, same as any real rating.
    """
    for i in range(n_per_group):
        uid = f'group_a_{i}'
        ratings_service.save_rating(uid, 'squat', rng.randint(4, 5))
        ratings_service.save_rating(uid, 'lunge', rng.randint(4, 5))
        ratings_service.save_rating(uid, 'biceps_curl', rng.randint(1, 3))
        ratings_service.save_rating(uid, 'shoulder_press', rng.randint(1, 3))
    for i in range(n_per_group):
        uid = f'group_b_{i}'
        ratings_service.save_rating(uid, 'squat', rng.randint(1, 3))
        ratings_service.save_rating(uid, 'lunge', rng.randint(1, 3))
        ratings_service.save_rating(uid, 'biceps_curl', rng.randint(4, 5))
        ratings_service.save_rating(uid, 'shoulder_press', rng.randint(4, 5))


def _seed_session_score(db, user_id: str, exercise_id: str, overall_score: float) -> None:
    """Minimal session doc — just enough for scores_service to average
    overall_score per (user, exercise)."""
    db.sessions.insert_one({
        '_id': f'{user_id}-{exercise_id}-{overall_score}',
        'user_id': user_id, 'exercise_id': exercise_id, 'overall_score': overall_score,
    })


class TestRecommendForUser(MongoMockMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        _seed_correlated_population(random.Random(42))
        train()

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
        # The seeded population's "group A" rates squat/lunge high and
        # biceps_curl/shoulder_press low together — a user who loved squat
        # should get lunge predicted above the upper-body pair.
        results = recommend_for_user({'squat': 5}, CATALOG)
        by_id = {r.exercise.exercise_id: r.score for r in results}
        self.assertGreater(by_id['lunge'], by_id['biceps_curl'])
        self.assertGreater(by_id['lunge'], by_id['shoulder_press'])

    def test_unrated_exercises_are_predicted_from_ratings(self):
        results = recommend_for_user({'squat': 5}, CATALOG)
        for r in results:
            if r.exercise.exercise_id != 'squat':
                self.assertEqual(r.reason_code, 'predicted_from_ratings')


class TestNewRatingsInfluenceTraining(MongoMockMixin, unittest.TestCase):
    """New ratings written to Mongo must actually shape the learned
    weights on the next retrain, not just each rater's own prediction."""

    def setUp(self):
        super().setUp()
        _seed_correlated_population(random.Random(42))

    def test_new_ratings_shift_the_predicted_scores(self):
        # Predicting biceps_curl's score for a user who rated squat 5/5 —
        # checked at the prediction level rather than one internal weight,
        # since gradient descent can trade a shift off between a weight and
        # the bias term depending on how the dataset changed.
        biceps_curl = [e for e in CATALOG if e.exercise_id == 'biceps_curl']

        train()
        before = recommend_for_user({'squat': 5}, biceps_curl)[0].score

        # The seeded population makes squat and biceps_curl strongly
        # NEGATIVELY correlated. Add users who consistently break that
        # pattern (love both) and retrain — the predicted score for
        # biceps_curl (given squat=5) must rise, not stay pinned.
        for i in range(15):
            ratings_service.save_rating(f'new_{i}', 'squat', 5)
            ratings_service.save_rating(f'new_{i}', 'lunge', 5)
            ratings_service.save_rating(f'new_{i}', 'biceps_curl', 5)
            ratings_service.save_rating(f'new_{i}', 'shoulder_press', 5)

        train()
        after = recommend_for_user({'squat': 5}, biceps_curl)[0].score

        self.assertGreater(after, before)

    def test_new_ratings_are_included_in_training_pool(self):
        ratings_service.save_rating('someone', 'squat', 5)
        ratings_service.save_rating('someone', 'lunge', 4)
        pool = ratings_service.get_all_ratings_for_training()
        self.assertEqual(pool['someone'], {'squat': 5, 'lunge': 4})


class TestAvgScoreFeature(MongoMockMixin, unittest.TestCase):
    """The ranker's second feature — a user's own average session form
    score on a predictor exercise — must actually move predictions on its
    own, independent of the rating feature."""

    def test_predictor_avg_score_shifts_the_prediction(self):
        biceps_curl = [e for e in CATALOG if e.exercise_id == 'biceps_curl']

        # Every seeded user gives squat the SAME rating, but half perform
        # it well and rate biceps_curl high, half perform it poorly and
        # rate biceps_curl low — ties squat *performance* (not the rating,
        # which is identical for everyone) to the biceps_curl rating.
        for i in range(15):
            performs_well = i % 2 == 0
            ratings_service.save_rating(f'perf_{i}', 'squat', 3)
            ratings_service.save_rating(f'perf_{i}', 'biceps_curl', 5 if performs_well else 1)
            _seed_session_score(self._mock_db, f'perf_{i}', 'squat', 95.0 if performs_well else 20.0)

        train()

        # Same rating input for both — only the avg squat score differs.
        high_performer = recommend_for_user({'squat': 3}, biceps_curl, user_scores={'squat': 95.0})[0].score
        low_performer  = recommend_for_user({'squat': 3}, biceps_curl, user_scores={'squat': 20.0})[0].score

        self.assertGreater(high_performer, low_performer)


# ── Ratings storage ────────────────────────────────────────────────────────────

class TestRatingsService(MongoMockMixin, unittest.TestCase):

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

    def test_get_all_ratings_for_training_groups_one_dict_per_user(self):
        ratings_service.save_rating('u1', 'squat', 4)
        ratings_service.save_rating('u1', 'lunge', 2)
        ratings_service.save_rating('u2', 'squat', 5)
        pool = ratings_service.get_all_ratings_for_training()
        self.assertEqual(pool['u1'], {'squat': 4, 'lunge': 2})
        self.assertEqual(pool['u2'], {'squat': 5})


if __name__ == '__main__':
    unittest.main()
