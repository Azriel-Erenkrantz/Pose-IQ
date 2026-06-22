"""
Core recommendation logic.

Input:
  - CATALOG exercises (filtered to target region)
  - user's health status
  - user's exercise history (from the performance-monitoring model)
  - community records (other users with similar health profiles)
  - user's own past feedback

Output:
  - Ranked list of ExerciseRecommendation, sorted by score descending.

Scoring overview
----------------
Each exercise gets a personal_score derived from the user's history, shaped by
which health scenario applies (see HealthScenario).  A community_score and an
optional feedback_score are blended in with fixed weights.  When a signal is
missing (no history / no community match), its weight is redistributed to the
remaining signals.

Health scenarios
----------------
ALL_HEALTHY
    Rate high the exercises the user historically found difficult — now at full
    health is the right time to improve on weaknesses.

TARGET_HEALTHY_ADJACENT_NOT
    Prefer moderate-difficulty exercises (peak around 0.45 difficulty).  Anything
    historically very hard (>0.75) is penalised — we don't want to stress the
    adjacent region.

TARGET_UNHEALTHY
    Only recommend exercises with good form history and manageable difficulty.
    CRITICAL: if difficulty >= 0.90 AND the exercise targets an unhealthy region,
    score is near-zero — too risky while injured.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import (
    BodyRegion,
    CatalogExercise,
    CommunityRecord,
    ExercisePerformanceRecord,
    ExerciseRecommendation,
    HealthScenario,
    HealthStatus,
    UserFeedback,
)

HEALTHY_THRESHOLD = 4          # rating >= 4 counts as "healthy"
COMMUNITY_SIMILARITY_MIN = 2   # how many regions must be within ±1 to count as similar

ADJACENT: Dict[BodyRegion, List[BodyRegion]] = {
    BodyRegion.UPPER: [BodyRegion.CORE],
    BodyRegion.CORE:  [BodyRegion.UPPER, BodyRegion.LOWER],
    BodyRegion.LOWER: [BodyRegion.CORE],
}

WEIGHT_PERSONAL   = 0.60
WEIGHT_COMMUNITY  = 0.25
WEIGHT_FEEDBACK   = 0.15


# ── Scenario detection ──────────────────────────────────────────────────────

def detect_scenario(health: HealthStatus, target: BodyRegion) -> HealthScenario:
    if all(health.get(r) >= HEALTHY_THRESHOLD for r in BodyRegion):
        return HealthScenario.ALL_HEALTHY
    if health.get(target) < HEALTHY_THRESHOLD:
        return HealthScenario.TARGET_UNHEALTHY
    if any(health.get(r) < HEALTHY_THRESHOLD for r in ADJACENT[target]):
        return HealthScenario.TARGET_HEALTHY_ADJACENT_NOT
    return HealthScenario.ALL_HEALTHY


# ── Per-signal scores ───────────────────────────────────────────────────────

def _try_ml(
    exercise: CatalogExercise,
    history: List[ExercisePerformanceRecord],
    health: HealthStatus,
) -> Optional[float]:
    """
    Attempt ML scoring. Returns None on any failure so the caller falls back
    to rule-based scoring silently.
    Only fires for ALL_HEALTHY — safety-critical scenarios always use rules.
    """
    try:
        from .ml_ranker import has_enough_data, model_exists, predict
        if not (has_enough_data(history) and model_exists()):
            return None
        records = [r for r in history if r.exercise_id == exercise.exercise_id]
        return predict(records, health)
    except Exception:
        return None


def _personal_score(
    exercise: CatalogExercise,
    history: List[ExercisePerformanceRecord],
    scenario: HealthScenario,
    health: HealthStatus,
) -> Tuple[float, str]:
    # ML path — ALL_HEALTHY only; safety scenarios always use rule-based
    if scenario == HealthScenario.ALL_HEALTHY:
        ml = _try_ml(exercise, history, health)
        if ml is not None:
            return ml, f"ML satisfaction estimate: {ml:.0%}"

    records = [r for r in history if r.exercise_id == exercise.exercise_id]

    if not records:
        if scenario == HealthScenario.TARGET_UNHEALTHY:
            return 0.30, "No history — cannot assess safety while not at best"
        return 0.50, "No history — neutral score"

    recent = sorted(records, key=lambda r: r.timestamp, reverse=True)[:5]
    avg_diff = round(sum(r.difficulty_score for r in recent) / len(recent), 3)
    avg_form = round(sum(r.form_quality_score for r in recent) / len(recent), 3)

    if scenario == HealthScenario.ALL_HEALTHY:
        score = avg_diff
        return score, (
            f"Historically challenging ({avg_diff:.0%}) — "
            "ideal to tackle now at full health"
        )

    if scenario == HealthScenario.TARGET_HEALTHY_ADJACENT_NOT:
        # Score peaks at ~0.45 difficulty; high difficulty = bad with adjacent pain
        raw = 1.0 - abs(avg_diff - 0.45) * 2
        score = max(0.0, raw)
        if avg_diff > 0.75:
            return score, (
                f"Too challenging historically ({avg_diff:.0%}) — "
                "reduced score while adjacent region is limited"
            )
        return score, (
            f"Suitable difficulty ({avg_diff:.0%}) — "
            "appropriate given adjacent region issue"
        )

    # TARGET_UNHEALTHY
    affected = [r for r in BodyRegion if health.get(r) < HEALTHY_THRESHOLD]
    region_overlap = any(r in exercise.body_regions for r in affected)

    if region_overlap and avg_diff >= 0.90:
        return 0.05, (
            f"High difficulty ({avg_diff:.0%}) on an affected region — "
            "not safe while injured"
        )
    if avg_form < 0.60:
        return 0.10, (
            f"Poor form history ({avg_form:.0%}) — "
            "unsafe to practise while not at best"
        )
    score = (1.0 - avg_diff) * avg_form
    return round(score, 3), (
        f"Good form ({avg_form:.0%}), manageable difficulty ({avg_diff:.0%}) — "
        "safe to continue"
    )


def _community_score(
    exercise: CatalogExercise,
    community: List[CommunityRecord],
    health: HealthStatus,
) -> Optional[float]:
    matched = [
        c.user_rating for c in community
        if c.exercise_id == exercise.exercise_id
        and sum(
            1 for r in BodyRegion
            if abs(c.health_ratings.get(r, 3) - health.get(r)) <= 1
        ) >= COMMUNITY_SIMILARITY_MIN
    ]
    return round(sum(matched) / len(matched), 3) if matched else None


def _feedback_score(
    exercise: CatalogExercise,
    feedbacks: List[UserFeedback],
) -> Optional[float]:
    matched = [f.rating / 5.0 for f in feedbacks if f.exercise_id == exercise.exercise_id]
    return round(sum(matched) / len(matched), 3) if matched else None


def _blend(personal: float, community: Optional[float], feedback: Optional[float]) -> float:
    """Weighted average; redistributes weights when signals are absent."""
    components: List[Tuple[float, float]] = [(personal, WEIGHT_PERSONAL)]
    if community is not None:
        components.append((community, WEIGHT_COMMUNITY))
    if feedback is not None:
        components.append((feedback, WEIGHT_FEEDBACK))
    total_w = sum(w for _, w in components)
    return round(sum(s * w for s, w in components) / total_w, 3)


# ── Public API ──────────────────────────────────────────────────────────────

def recommend(
    exercises: List[CatalogExercise],
    target_region: BodyRegion,
    health: HealthStatus,
    history: List[ExercisePerformanceRecord],
    community: List[CommunityRecord],
    feedbacks: List[UserFeedback],
) -> List[ExerciseRecommendation]:
    """
    Returns exercises sorted by recommendation score (highest first).

    Step 1 (non-ML): filter to primary_region == target_region.
    Step 2 (ML):     score each exercise using personal + community + feedback signals.
    """
    filtered = [e for e in exercises if e.primary_region == target_region]
    scenario = detect_scenario(health, target_region)

    results: List[ExerciseRecommendation] = []
    for ex in filtered:
        p_score, reason = _personal_score(ex, history, scenario, health)
        c_score = _community_score(ex, community, health)
        f_score = _feedback_score(ex, feedbacks)
        final = _blend(p_score, c_score, f_score)

        results.append(ExerciseRecommendation(
            exercise=ex,
            score=final,
            reason=reason,
            scenario=scenario,
            personal_score=p_score,
            community_score=c_score,
            feedback_score=f_score,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
