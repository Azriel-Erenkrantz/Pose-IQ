"""
Progressive-overload weight recommendations.

Double-progression adapted to the data Pose-IQ actually has: instead of
"reps in reserve" (which a camera can't see), progression is gated on the
form score the pipeline already computes per session.

The rule, per exercise:
  1. No weight ever logged           → start light / bodyweight, log weights.
  2. Last session's form broke down  → back off one increment.
  3. Enough consecutive sessions at the current weight with clean form
     (threshold depends on fitness level) → add one increment.
  4. Otherwise                       → hold the weight, consolidate form.

Consumes app_model types only (LiveSessionOutput in, WeightRecommendation
out) so the API can call it directly on what service.get_history returns.
"""
from __future__ import annotations

from typing import Dict, List

from core.app_model import (
    FitnessLevel,
    LiveSessionOutput,
    User,
    WeightRecommendation,
)

# Weight jump per progression step. Lower-body moves tolerate bigger jumps.
INCREMENT_KG: Dict[str, float] = {
    'squat':          5.0,
    'lunge':          2.0,
    'biceps_curl':    2.0,
    'shoulder_press': 2.0,
}
DEFAULT_INCREMENT_KG = 2.0

FORM_READY_SCORE     = 85.0   # a session at/above this counts toward progression
FORM_BREAKDOWN_SCORE = 70.0   # a session below this triggers a back-off

# Clean sessions at the current weight required to progress.
SESSIONS_TO_PROGRESS: Dict[FitnessLevel, int] = {
    FitnessLevel.BEGINNER:     3,
    FitnessLevel.INTERMEDIATE: 2,
    FitnessLevel.ADVANCED:     2,
}


def recommend_weight(
    exercise_id: str,
    exercise_name: str,
    sessions: List[LiveSessionOutput],
    fitness_level: FitnessLevel = FitnessLevel.INTERMEDIATE,
) -> WeightRecommendation:
    """Recommend the next working weight for one exercise.

    `sessions` may contain any exercises in any order; only this exercise's
    sessions are considered, newest first.
    """
    increment = INCREMENT_KG.get(exercise_id, DEFAULT_INCREMENT_KG)
    ex_sessions = sorted(
        (s for s in sessions if s.exercise_id == exercise_id),
        key=lambda s: s.date, reverse=True,
    )
    logged = [s for s in ex_sessions if s.weight_kg is not None]

    if not logged:
        return WeightRecommendation(
            exercise_id           = exercise_id,
            exercise_name         = exercise_name,
            recommended_weight_kg = 0.0,
            reasoning             = ('No weight logged yet — start with bodyweight '
                                     'or a light load and log it after each session.'),
            confidence            = 0.2,
            reasoning_code        = 'no_weight_logged',
        )

    current = logged[0].weight_kg
    # Most recent consecutive sessions at the current weight — earlier
    # sessions at other weights say nothing about readiness at this one.
    streak: List[LiveSessionOutput] = []
    for s in logged:
        if s.weight_kg != current:
            break
        streak.append(s)

    needed = SESSIONS_TO_PROGRESS[fitness_level]

    if streak[0].overall_score < FORM_BREAKDOWN_SCORE:
        return WeightRecommendation(
            exercise_id           = exercise_id,
            exercise_name         = exercise_name,
            recommended_weight_kg = max(0.0, current - increment),
            reasoning             = (f'Form score dropped to {streak[0].overall_score:.0f} '
                                     f'at {current:g} kg — reduce the load and rebuild '
                                     'clean technique.'),
            confidence            = 0.8,
            reasoning_code        = 'form_dropped',
            reasoning_params      = {'score': streak[0].overall_score, 'weight': current},
        )

    clean = [s for s in streak if s.overall_score >= FORM_READY_SCORE]
    if len(clean) >= needed:
        return WeightRecommendation(
            exercise_id           = exercise_id,
            exercise_name         = exercise_name,
            recommended_weight_kg = current + increment,
            reasoning             = (f'{len(clean)} sessions at {current:g} kg with form '
                                     f'≥ {FORM_READY_SCORE:.0f} — ready to add '
                                     f'{increment:g} kg.'),
            confidence            = min(0.9, 0.6 + 0.1 * len(clean)),
            reasoning_code        = 'ready_to_increase',
            reasoning_params      = {'clean': len(clean), 'weight': current,
                                      'threshold': FORM_READY_SCORE, 'increment': increment},
        )

    return WeightRecommendation(
        exercise_id           = exercise_id,
        exercise_name         = exercise_name,
        recommended_weight_kg = current,
        reasoning             = (f'{len(clean)}/{needed} clean sessions at {current:g} kg '
                                 f'(form ≥ {FORM_READY_SCORE:.0f}) — stay here until '
                                 'the technique is consistent.'),
        confidence            = 0.6,
        reasoning_code        = 'stay_here',
        reasoning_params      = {'clean': len(clean), 'needed': needed, 'weight': current,
                                  'threshold': FORM_READY_SCORE},
    )


def recommend_weights_for_user(
    user: User,
    sessions: List[LiveSessionOutput],
) -> List[WeightRecommendation]:
    """One recommendation per exercise the user has trained, most recently
    trained exercise first."""
    seen: Dict[str, str] = {}
    for s in sorted(sessions, key=lambda s: s.date, reverse=True):
        if s.exercise_id not in seen:
            seen[s.exercise_id] = s.exercise_name
    return [
        recommend_weight(ex_id, ex_name, sessions, user.fitness_level)
        for ex_id, ex_name in seen.items()
    ]
