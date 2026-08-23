"""
Integration adapter: connects app_model types to the recommendation engine.

Entry point:
  recommend_for_user()  — call from the API; takes User + HealthStatus + session history

from_live_session() is the internal converter used by recommend_for_user().
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from ..app_model import (
    BodyRegion,
    CommunityRecord,
    Equipment,
    Exercise,
    ExercisePerformanceRecord,
    ExerciseRecommendation,
    HealthStatus,
    LiveSessionOutput,
    RecommendationSession,
    TargetGoal,
    User,
    UserFeedback,
)
from .catalog import CATALOG, COMMUNITY_DATA
from .recommender import recommend

# Maps pipeline joint names → body regions for limitation filtering.
_JOINT_TO_REGION: Dict[str, BodyRegion] = {
    'spine':          BodyRegion.CORE,
    'right_knee':     BodyRegion.LOWER,
    'left_knee':      BodyRegion.LOWER,
    'right_arm_body': BodyRegion.UPPER,
    'left_arm_body':  BodyRegion.UPPER,
    'right_elbow':    BodyRegion.UPPER,
    'left_elbow':     BodyRegion.UPPER,
}

# Maps user goal → body region to train. First matching goal wins.
_GOAL_TO_REGION: Dict[TargetGoal, BodyRegion] = {
    TargetGoal.LEGS:      BodyRegion.LOWER,
    TargetGoal.CARDIO:    BodyRegion.LOWER,
    TargetGoal.ABS:       BodyRegion.CORE,
    TargetGoal.ARMS:      BodyRegion.UPPER,
    TargetGoal.FULL_BODY: BodyRegion.LOWER,
    TargetGoal.OTHER:     BodyRegion.LOWER,
}


def _region_for_goals(goals: List[TargetGoal]) -> BodyRegion:
    for g in goals:
        if g in _GOAL_TO_REGION:
            return _GOAL_TO_REGION[g]
    return BodyRegion.LOWER


def _has_equipment(user_equipment: List[Equipment], required: List[Equipment]) -> bool:
    """True when the user owns at least one of the required equipment types, or none required."""
    if not required:
        return True
    return any(eq in user_equipment for eq in required)


def recommend_for_user(
    user: User,
    health: HealthStatus,
    sessions: List[LiveSessionOutput],
    target_region: Optional[BodyRegion] = None,
    community: Optional[List[CommunityRecord]] = None,
    feedbacks: Optional[List[UserFeedback]] = None,
) -> List[ExerciseRecommendation]:
    """
    Main entry point for the API.

    - Derives target_region from user's goals when not provided.
    - Excludes exercises the user can't do (equipment, limitations).
    - Converts session history to performance records.
    - Delegates scoring to recommend().
    """
    if target_region is None:
        target_region = _region_for_goals(user.target_goals)

    history = [from_live_session(s) for s in sessions]

    limited_regions = {
        _JOINT_TO_REGION[j]
        for j in user.limited_joints
        if j in _JOINT_TO_REGION
    }

    exercises = [
        e for e in CATALOG
        if e.primary_region not in limited_regions
        and _has_equipment(user.equipment, e.equipment_required)
    ]

    return recommend(
        exercises=exercises,
        target_region=target_region,
        health=health,
        history=history,
        community=community if community is not None else COMMUNITY_DATA,
        feedbacks=feedbacks or [],
    )


def from_live_session(
    session: LiveSessionOutput,
    health_snapshot: Optional[Dict[BodyRegion, int]] = None,
) -> ExercisePerformanceRecord:
    """Convert a LiveSessionOutput (app_model) → ExercisePerformanceRecord."""
    form = round(session.overall_score / 100.0, 3)
    violations = list({j for rep in session.reps for j in rep.error_joints})
    return ExercisePerformanceRecord(
        id=str(uuid.uuid4()),
        exercise_id=session.exercise_id,
        user_id=session.user_id,
        session_id=session.session_id,
        timestamp=session.date,
        difficulty_score=round(1.0 - form, 3),
        form_quality_score=form,
        completion_rate=1.0,
        violations=violations,
        reps_completed=session.total_reps,
        sets_completed=1,
        health_snapshot=health_snapshot or {r: 3 for r in BodyRegion},
    )
