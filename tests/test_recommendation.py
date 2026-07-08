"""
Modularity tests for the recommendation engine.

Run from project root:
    python3 -m unittest core.recommendation.tests          # full suite
    python3 -m unittest core.recommendation.tests.TestModularity  # one class

Each TestCase targets one modular piece independently:
    TestScenarioDetection — health + adjacency logic
    TestPersonalScoring   — per-scenario scoring from performance history
    TestCatalogFilter     — region filtering (non-ML step)
    TestBlend             — signal weighting / fallback when signals are absent
    TestModularity        — prove each piece is swappable without touching others
"""
from __future__ import annotations

import sys
import os
import unittest
from datetime import datetime

# Allow running directly as a script: python3 core/recommendation/tests.py
_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.app_model import (
    BodyRegion, Exercise, CommunityRecord,
    ExercisePerformanceRecord, HealthScenario, HealthStatus, UserFeedback,
)
from core.recommendation.recommender import (
    ADJACENT, detect_scenario, recommend, _variety_multiplier,
)
from core.recommendation.catalog import CATALOG, COMMUNITY_DATA
from core.recommendation.fake_data import (
    scenario_all_healthy,
    scenario_core_pain,
    scenario_upper_injured,
    scenario_new_user,
)

U = BodyRegion.UPPER
C = BodyRegion.CORE
L = BodyRegion.LOWER


# ── helpers ───────────────────────────────────────────────────────────────────

def _health(u=5, c=5, lo=5) -> HealthStatus:
    return HealthStatus("_t", {U: u, C: c, L: lo})


def _rec_by_id(recs, exercise_id):
    return next(r for r in recs if r.exercise.exercise_id == exercise_id)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenario detection
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioDetection(unittest.TestCase):

    def test_all_ratings_at_threshold_is_healthy(self):
        self.assertEqual(detect_scenario(_health(5, 5, 5), U), HealthScenario.ALL_HEALTHY)

    def test_target_region_below_threshold_is_unhealthy(self):
        self.assertEqual(detect_scenario(_health(u=2, c=5, lo=5), U), HealthScenario.TARGET_UNHEALTHY)

    def test_adjacent_below_threshold_triggers_adjacent_scenario(self):
        # Core is adjacent to upper; core pain while upper is healthy
        self.assertEqual(
            detect_scenario(_health(u=5, c=2, lo=5), U),
            HealthScenario.TARGET_HEALTHY_ADJACENT_NOT,
        )

    def test_core_is_adjacent_to_lower(self):
        self.assertEqual(
            detect_scenario(_health(u=5, c=2, lo=5), L),
            HealthScenario.TARGET_HEALTHY_ADJACENT_NOT,
        )

    def test_non_adjacent_region_does_not_affect_scenario(self):
        # Lower pain does not affect upper training (they are not adjacent)
        self.assertEqual(detect_scenario(_health(u=5, c=5, lo=2), U), HealthScenario.ALL_HEALTHY)

    def test_boundary_rating_exactly_at_threshold_is_healthy(self):
        import core.recommendation.recommender as m
        # Rating == HEALTHY_THRESHOLD should count as healthy
        self.assertEqual(
            detect_scenario(_health(m.HEALTHY_THRESHOLD, m.HEALTHY_THRESHOLD, m.HEALTHY_THRESHOLD), U),
            HealthScenario.ALL_HEALTHY,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Personal scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonalScoring(unittest.TestCase):

    def test_all_healthy_hard_exercise_scores_higher_than_easy(self):
        sc = scenario_all_healthy()
        recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        sp = _rec_by_id(recs, "shoulder_press")   # avg_diff ≈ 0.815
        bc = _rec_by_id(recs, "bicep_curl")       # avg_diff ≈ 0.290
        self.assertGreater(sp.personal_score, bc.personal_score)

    def test_core_pain_hard_exercise_penalised(self):
        sc = scenario_core_pain()
        recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        sp = _rec_by_id(recs, "shoulder_press")   # avg_diff ≈ 0.82 → penalised
        pu = _rec_by_id(recs, "push_up")          # avg_diff ≈ 0.44 → near optimum
        self.assertGreater(pu.personal_score, sp.personal_score)

    def test_upper_injured_dangerous_exercise_near_zero(self):
        sc = scenario_upper_injured()
        recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        bc = _rec_by_id(recs, "bicep_curl")       # diff=0.92 on injured region
        self.assertLessEqual(bc.personal_score, 0.10)

    def test_upper_injured_safe_exercise_scores_high(self):
        sc = scenario_upper_injured()
        recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        pu = _rec_by_id(recs, "push_up")          # diff=0.29, form=0.91
        self.assertGreater(pu.personal_score, 0.50)

    def test_no_history_neutral_score(self):
        sc = scenario_new_user()
        recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        for rec in recs:
            self.assertEqual(rec.personal_score, 0.50)

    def test_injured_no_history_conservative_score(self):
        # No history + TARGET_UNHEALTHY → 0.30 (can't assess safety)
        health = HealthStatus("u", {U: 2, C: 5, L: 5})
        recs = recommend(CATALOG, U, health, [], [], [])
        for rec in recs:
            self.assertEqual(rec.personal_score, 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Catalog filter
# ─────────────────────────────────────────────────────────────────────────────

class TestCatalogFilter(unittest.TestCase):

    def test_filter_returns_only_target_region(self):
        health = HealthStatus("u", {U: 5, C: 5, L: 5})
        for region in BodyRegion:
            recs = recommend(CATALOG, region, health, [], [], [])
            for rec in recs:
                self.assertEqual(
                    rec.exercise.primary_region, region,
                    f"Expected {region}, got {rec.exercise.primary_region} for {rec.exercise.name}",
                )

    def test_custom_catalog_is_fully_isolated(self):
        # Pass a 2-exercise catalog; scoring logic should work without knowing about it
        mini = [
            Exercise("ex_a", "Alpha", U, [U], 0.3, "test"),
            Exercise("ex_b", "Beta",  U, [U], 0.7, "test"),
        ]
        sc = scenario_all_healthy()
        recs = recommend(mini, U, sc.health, sc.history, [], [])
        self.assertEqual(len(recs), 2)
        self.assertEqual({r.exercise.exercise_id for r in recs}, {"ex_a", "ex_b"})

    def test_adding_exercise_to_catalog_includes_it(self):
        new_ex = Exercise("new_upper", "New Move", U, [U], 0.5, "test")
        extended = CATALOG + [new_ex]
        sc = scenario_new_user()
        recs = recommend(extended, U, sc.health, sc.history, [], [])
        ids = [r.exercise.exercise_id for r in recs]
        self.assertIn("new_upper", ids)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Signal blending
# ─────────────────────────────────────────────────────────────────────────────

class TestBlend(unittest.TestCase):

    def test_no_community_no_feedback_score_equals_personal_times_variety(self):
        # Without community or feedback, score = personal_score × variety_multiplier
        sc = scenario_all_healthy()
        recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        for rec in recs:
            if rec.community_score is None and rec.feedback_score is None:
                variety = _variety_multiplier(rec.exercise, sc.history)
                self.assertEqual(rec.score, round(rec.personal_score * variety, 3))

    def test_community_signal_changes_final_score(self):
        sc = scenario_all_healthy()
        # push_up has community records in COMMUNITY_DATA; compare with/without
        recs_no   = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        recs_comm = recommend(CATALOG, sc.target_region, sc.health, sc.history, COMMUNITY_DATA, [])
        pu_no   = _rec_by_id(recs_no,   "push_up")
        pu_comm = _rec_by_id(recs_comm, "push_up")
        self.assertIsNone(pu_no.community_score)
        self.assertIsNotNone(pu_comm.community_score)
        self.assertNotEqual(pu_no.score, pu_comm.score)

    def test_high_community_rating_raises_score(self):
        high_comm = [
            CommunityRecord("1", "push_up", {U: 5, C: 5, L: 5}, 0.95, datetime.now()),
        ]
        low_comm = [
            CommunityRecord("2", "push_up", {U: 5, C: 5, L: 5}, 0.05, datetime.now()),
        ]
        sc = scenario_all_healthy()
        recs_high = recommend(CATALOG, sc.target_region, sc.health, sc.history, high_comm, [])
        recs_low  = recommend(CATALOG, sc.target_region, sc.health, sc.history, low_comm, [])
        pu_high = _rec_by_id(recs_high, "push_up")
        pu_low  = _rec_by_id(recs_low,  "push_up")
        # Personal scores identical; community drives the difference
        self.assertEqual(pu_high.personal_score, pu_low.personal_score)
        self.assertGreater(pu_high.score, pu_low.score)

    def test_feedback_signal_changes_final_score(self):
        sc = scenario_all_healthy()
        feedbacks = [
            UserFeedback("f1", "push_up", sc.user_id, "sess", datetime.now(), rating=5),
            UserFeedback("f2", "push_up", sc.user_id, "sess", datetime.now(), rating=5),
        ]
        recs_no = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
        recs_fb = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], feedbacks)
        pu_no = _rec_by_id(recs_no, "push_up")
        pu_fb = _rec_by_id(recs_fb, "push_up")
        self.assertIsNone(pu_no.feedback_score)
        self.assertIsNotNone(pu_fb.feedback_score)
        self.assertNotEqual(pu_no.score, pu_fb.score)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Modularity — swap individual pieces, verify only that piece changes
# ─────────────────────────────────────────────────────────────────────────────

class TestModularity(unittest.TestCase):

    def test_swap_catalog_scoring_logic_unchanged(self):
        """Replace the entire catalog; scoring logic runs identically on the new exercises."""
        custom = [
            Exercise("hard_ex",  "Hard",  U, [U], 0.9, "test"),
            Exercise("easy_ex",  "Easy",  U, [U], 0.1, "test"),
        ]
        sc = scenario_all_healthy()
        # Seed history for the custom exercises at the same difficulty pattern
        from core.recommendation.fake_data import _rec
        h = {U: 5, C: 5, L: 5}
        custom_history = [
            _rec(sc.user_id, "hard_ex", 0.82, 0.60, [], h),
            _rec(sc.user_id, "easy_ex", 0.28, 0.92, [], h),
        ]
        recs = recommend(custom, U, sc.health, custom_history, [], [])
        hard = _rec_by_id(recs, "hard_ex")
        easy = _rec_by_id(recs, "easy_ex")
        # ALL_HEALTHY: hard exercise should rank higher
        self.assertGreater(hard.score, easy.score)

    def test_swap_adjacent_map_changes_scenario_only(self):
        """Rewire the ADJACENT map; scenario detection changes, scoring formula stays."""
        import core.recommendation.recommender as m
        original = {k: list(v) for k, v in m.ADJACENT.items()}
        try:
            m.ADJACENT[U] = []           # upper now has no adjacent regions
            h = _health(u=5, c=2, lo=5)  # core has pain, but not adjacent to upper anymore
            self.assertEqual(detect_scenario(h, U), HealthScenario.ALL_HEALTHY)
        finally:
            m.ADJACENT.update(original)
        # After restore, original behaviour returns
        self.assertEqual(detect_scenario(_health(u=5, c=2, lo=5), U),
                         HealthScenario.TARGET_HEALTHY_ADJACENT_NOT)

    def test_swap_healthy_threshold_changes_scenario_boundaries(self):
        """Lower HEALTHY_THRESHOLD; a rating=3 now counts as healthy."""
        import core.recommendation.recommender as m
        original = m.HEALTHY_THRESHOLD
        h = _health(u=5, c=3, lo=5)
        try:
            m.HEALTHY_THRESHOLD = 4
            self.assertEqual(detect_scenario(h, U), HealthScenario.TARGET_HEALTHY_ADJACENT_NOT)
            m.HEALTHY_THRESHOLD = 3
            self.assertEqual(detect_scenario(h, U), HealthScenario.ALL_HEALTHY)
        finally:
            m.HEALTHY_THRESHOLD = original

    def test_community_data_independent_from_personal_score(self):
        """Swapping community data changes community_score but never personal_score."""
        sc = scenario_all_healthy()
        high = [CommunityRecord("1", "push_up", {U: 5, C: 5, L: 5}, 0.99, datetime.now())]
        low  = [CommunityRecord("2", "push_up", {U: 5, C: 5, L: 5}, 0.01, datetime.now())]

        recs_h = recommend(CATALOG, sc.target_region, sc.health, sc.history, high, [])
        recs_l = recommend(CATALOG, sc.target_region, sc.health, sc.history, low,  [])
        pu_h = _rec_by_id(recs_h, "push_up")
        pu_l = _rec_by_id(recs_l, "push_up")

        self.assertEqual(pu_h.personal_score, pu_l.personal_score,
                         "personal_score must not change when community data changes")
        self.assertGreater(pu_h.score, pu_l.score,
                           "final score must reflect the different community signals")

    def test_empty_history_does_not_crash_any_scenario(self):
        """All three scenarios handle empty history gracefully."""
        for health_vals, target in [
            ({U: 5, C: 5, L: 5}, U),   # ALL_HEALTHY
            ({U: 5, C: 2, L: 5}, U),   # TARGET_HEALTHY_ADJACENT_NOT
            ({U: 2, C: 5, L: 5}, U),   # TARGET_UNHEALTHY
        ]:
            health = HealthStatus("u", health_vals)
            recs = recommend(CATALOG, target, health, [], [], [])
            self.assertTrue(len(recs) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ML Ranker
# ─────────────────────────────────────────────────────────────────────────────

import tempfile
import uuid
from datetime import timedelta

from core.recommendation.recommender import _variety_multiplier, VARIETY_LOOKBACK, VARIETY_PENALTY

from core.recommendation.ml_ranker import (
    MIN_SESSIONS_TOTAL, _feature_vector, _satisfaction_score,
    has_enough_data, model_exists, train, predict,
)


def _make_records(
    user_id: str,
    exercise_id: str,
    health_map: dict,
    n: int,
    difficulty: float,
    form: float,
    completion: float,
    form_trend: float = 0.0,   # added per-session to form to simulate improvement
) -> list:
    records = []
    for i in range(n):
        records.append(ExercisePerformanceRecord(
            id=str(uuid.uuid4()),
            exercise_id=exercise_id,
            user_id=user_id,
            session_id=f"s{i}",
            timestamp=datetime.now() - timedelta(days=n - i),
            difficulty_score=difficulty,
            form_quality_score=min(1.0, form + form_trend * i),
            completion_rate=completion,
            violations=[],
            reps_completed=8,
            sets_completed=3,
            health_snapshot=health_map,
        ))
    return records


class TestVariety(unittest.TestCase):

    def _history_of(self, exercise_ids):
        """Build a fake history with one record per exercise_id in order (oldest first)."""
        health = {U: 5, C: 5, L: 5}
        records = []
        for i, ex_id in enumerate(exercise_ids):
            records.append(ExercisePerformanceRecord(
                id=str(uuid.uuid4()),
                exercise_id=ex_id,
                user_id="u",
                session_id=f"s{i}",
                timestamp=datetime.now() - timedelta(days=len(exercise_ids) - i),
                difficulty_score=0.5,
                form_quality_score=0.8,
                completion_rate=1.0,
                violations=[],
                reps_completed=8,
                sets_completed=3,
                health_snapshot=health,
            ))
        return records

    def _ex(self, exercise_id):
        return Exercise(exercise_id, exercise_id, U, [U], 0.5, "test")

    # ── Multiplier function ───────────────────────────────────────────────────

    def test_no_penalty_when_not_done_recently(self):
        history = self._history_of(["push_up", "bicep_curl", "tricep_dip"])
        self.assertEqual(_variety_multiplier(self._ex("shoulder_press"), history), 1.0)

    def test_no_penalty_when_no_history(self):
        self.assertEqual(_variety_multiplier(self._ex("push_up"), []), 1.0)

    def test_full_penalty_when_done_every_recent_session(self):
        history = self._history_of(["push_up"] * VARIETY_LOOKBACK)
        expected = round(1.0 - VARIETY_PENALTY, 3)
        self.assertEqual(_variety_multiplier(self._ex("push_up"), history), expected)

    def test_partial_penalty_proportional_to_frequency(self):
        # done 1 out of VARIETY_LOOKBACK recent sessions
        history = self._history_of(
            ["push_up"] + ["bicep_curl"] * (VARIETY_LOOKBACK - 1)
        )
        expected = round(1.0 - VARIETY_PENALTY * (1 / VARIETY_LOOKBACK), 3)
        self.assertEqual(_variety_multiplier(self._ex("push_up"), history), expected)

    def test_only_lookback_window_is_considered(self):
        # push_up done many times long ago but not in the recent window
        old = self._history_of(["push_up"] * 10)
        for r in old:
            r.timestamp = datetime.now() - timedelta(days=100)
        recent = self._history_of(["bicep_curl"] * VARIETY_LOOKBACK)
        self.assertEqual(_variety_multiplier(self._ex("push_up"), old + recent), 1.0)

    # ── Effect on final recommendation score ─────────────────────────────────

    def test_overused_exercise_scores_lower_than_fresh_alternative(self):
        health = HealthStatus("u", {U: 5, C: 5, L: 5})
        # push_up done every recent session; shoulder_press not done at all
        history = self._history_of(["push_up"] * VARIETY_LOOKBACK)
        recs = recommend(CATALOG, U, health, history, [], [])
        push_up        = _rec_by_id(recs, "push_up")
        shoulder_press = _rec_by_id(recs, "shoulder_press")
        # shoulder_press has no history (personal_score=0.5, variety=1.0)
        # push_up has no history either but variety multiplier < 1.0
        self.assertGreater(shoulder_press.score, push_up.score)

    def test_variety_does_not_penalise_exercises_done_in_different_region(self):
        health = HealthStatus("u", {U: 5, C: 5, L: 5})
        # squat (LOWER) done every session — should not penalise UPPER exercises
        history = self._history_of(["squat"] * VARIETY_LOOKBACK)
        recs = recommend(CATALOG, U, health, history, [], [])
        for rec in recs:
            self.assertEqual(rec.score, rec.personal_score,
                             f"{rec.exercise.name} should have no variety penalty")

    def test_variety_multiplier_is_swappable(self):
        """VARIETY_PENALTY is monkeypatchable — changing it affects scores immediately."""
        import core.recommendation.recommender as m
        original = m.VARIETY_PENALTY
        try:
            m.VARIETY_PENALTY = 0.0   # disable variety entirely
            history = self._history_of(["push_up"] * VARIETY_LOOKBACK)
            self.assertEqual(_variety_multiplier(self._ex("push_up"), history), 1.0)
        finally:
            m.VARIETY_PENALTY = original


class TestMLRanker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix='.pkl')

    def tearDown(self):
        import os
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    # ── Gate ─────────────────────────────────────────────────────────────────

    def test_insufficient_data_returns_false(self):
        self.assertFalse(has_enough_data([]))

    def test_sufficient_data_returns_true(self):
        health = {U: 5, C: 5, L: 5}
        records = _make_records("u", "ex", health, MIN_SESSIONS_TOTAL, 0.5, 0.8, 1.0)
        self.assertTrue(has_enough_data(records))

    def test_one_below_threshold_returns_false(self):
        health = {U: 5, C: 5, L: 5}
        records = _make_records("u", "ex", health, MIN_SESSIONS_TOTAL - 1, 0.5, 0.8, 1.0)
        self.assertFalse(has_enough_data(records))

    # ── Feature vector ────────────────────────────────────────────────────────

    def test_feature_vector_none_when_no_records(self):
        health = HealthStatus("u", {U: 5, C: 5, L: 5})
        self.assertIsNone(_feature_vector([], health))

    def test_feature_vector_correct_length(self):
        health_map = {U: 5, C: 5, L: 5}
        health = HealthStatus("u", health_map)
        records = _make_records("u", "ex", health_map, 3, 0.5, 0.8, 1.0)
        features = _feature_vector(records, health)
        self.assertEqual(len(features), 11)

    def test_feature_vector_health_normalised(self):
        health_map = {U: 5, C: 5, L: 5}
        health = HealthStatus("u", health_map)
        records = _make_records("u", "ex", health_map, 3, 0.5, 0.8, 1.0)
        features = _feature_vector(records, health)
        # Last three features are health values / 5.0 — all should be 1.0
        self.assertEqual(features[-3], 1.0)
        self.assertEqual(features[-2], 1.0)
        self.assertEqual(features[-1], 1.0)

    # ── Satisfaction label ────────────────────────────────────────────────────

    def test_satisfaction_none_when_no_records(self):
        self.assertIsNone(_satisfaction_score([], []))

    def test_high_form_high_completion_high_satisfaction(self):
        health = {U: 5, C: 5, L: 5}
        records = _make_records("u", "ex", health, 5, 0.5, 0.92, 1.0, form_trend=0.01)
        score = _satisfaction_score(records, [])
        self.assertGreater(score, 0.65)

    def test_low_form_low_completion_low_satisfaction(self):
        health = {U: 5, C: 5, L: 5}
        records = _make_records("u", "ex", health, 2, 0.5, 0.40, 0.40)
        score = _satisfaction_score(records, [])
        self.assertLess(score, 0.55)

    # ── Training and prediction ───────────────────────────────────────────────

    def test_train_succeeds_with_enough_exercises(self):
        health = {U: 5, C: 5, L: 5}
        history = (
            _make_records("u", "ex_a", health, 5, 0.5, 0.90, 1.0, 0.01) +
            _make_records("u", "ex_b", health, 5, 0.5, 0.40, 0.50, -0.01) +
            _make_records("u", "ex_c", health, 5, 0.6, 0.70, 0.80, 0.00)
        )
        result = train(history, [], model_path=self.tmp)
        self.assertTrue(result)
        self.assertTrue(model_exists(self.tmp))

    def test_train_fails_with_too_few_exercises(self):
        health = {U: 5, C: 5, L: 5}
        history = _make_records("u", "only_one", health, 5, 0.5, 0.8, 1.0)
        result = train(history, [], model_path=self.tmp)
        self.assertFalse(result)

    def test_predict_none_when_no_exercise_history(self):
        health_map = {U: 5, C: 5, L: 5}
        health = HealthStatus("u", health_map)
        all_history = (
            _make_records("u", "ex_a", health_map, 5, 0.5, 0.90, 1.0) +
            _make_records("u", "ex_b", health_map, 5, 0.5, 0.40, 0.50) +
            _make_records("u", "ex_c", health_map, 5, 0.6, 0.70, 0.80)
        )
        train(all_history, [], model_path=self.tmp)
        # Predict for exercise with no history
        result = predict([], health, model_path=self.tmp)
        self.assertIsNone(result)

    def test_model_ranks_high_satisfaction_above_low(self):
        """The model must learn that better form + more sessions = higher score."""
        health_map = {U: 5, C: 5, L: 5}
        health = HealthStatus("u", health_map)

        good = _make_records("u", "good_ex", health_map, 8, 0.5, 0.90, 1.00, form_trend=0.01)
        bad  = _make_records("u", "bad_ex",  health_map, 2, 0.5, 0.35, 0.40, form_trend=-0.02)
        mid  = _make_records("u", "mid_ex",  health_map, 4, 0.5, 0.65, 0.75)

        train(good + bad + mid, [], model_path=self.tmp)

        good_score = predict(good, health, model_path=self.tmp)
        bad_score  = predict(bad,  health, model_path=self.tmp)

        self.assertIsNotNone(good_score)
        self.assertIsNotNone(bad_score)
        self.assertGreater(good_score, bad_score)

    def test_score_clamped_between_0_and_1(self):
        health_map = {U: 5, C: 5, L: 5}
        health = HealthStatus("u", health_map)
        records_a = _make_records("u", "ex_a", health_map, 5, 0.5, 0.90, 1.0)
        records_b = _make_records("u", "ex_b", health_map, 5, 0.9, 0.20, 0.3)
        records_c = _make_records("u", "ex_c", health_map, 5, 0.6, 0.60, 0.7)
        train(records_a + records_b + records_c, [], model_path=self.tmp)
        for records in [records_a, records_b, records_c]:
            score = predict(records, health, model_path=self.tmp)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    # ── Integration with recommender ──────────────────────────────────────────

    def test_recommender_uses_ml_when_model_exists(self):
        """When model exists + enough data, reason string comes from ML path."""
        import core.recommendation.ml_ranker as m
        original_path = m.MODEL_PATH
        try:
            m.MODEL_PATH = self.tmp

            health_map = {U: 5, C: 5, L: 5}
            health = HealthStatus("u", health_map)

            # Build enough history across multiple UPPER exercises
            history = (
                _make_records("u", "push_up",        health_map, 8, 0.45, 0.85, 1.0, 0.01) +
                _make_records("u", "shoulder_press",  health_map, 8, 0.80, 0.55, 0.8) +
                _make_records("u", "bicep_curl",      health_map, 8, 0.28, 0.92, 1.0) +
                _make_records("u", "tricep_dip",      health_map, 2, 0.50, 0.70, 0.9)
            )
            train(history, [], model_path=self.tmp)

            recs = recommend(CATALOG, U, health, history, [], [])
            ml_recs = [r for r in recs if "ML" in r.reason]
            self.assertTrue(len(ml_recs) > 0, "Expected at least one ML-scored recommendation")
        finally:
            m.MODEL_PATH = original_path

    def test_recommender_falls_back_when_no_model(self):
        """Without a model file, reason string comes from rule-based path."""
        import core.recommendation.ml_ranker as m
        original_path = m.MODEL_PATH
        try:
            m.MODEL_PATH = '/tmp/nonexistent_ranker.pkl'
            sc = scenario_all_healthy()
            recs = recommend(CATALOG, sc.target_region, sc.health, sc.history, [], [])
            for rec in recs:
                self.assertNotIn("ML", rec.reason)
        finally:
            m.MODEL_PATH = original_path


if __name__ == '__main__':
    unittest.main(verbosity=2)
