"""
Integration adapter: connects app_model types to the recommendation engine.

Two entry points:
  recommend_for_user()  — call from the API; takes User + HealthStatus + session history
  from_pipeline_session() — call from the CLI pipeline after a workout session

from_live_session() is the internal converter used by recommend_for_user().
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..app_model import (
    BodyRegion,
    CommunityRecord,
    Exercise,
    ExercisePerformanceRecord,
    ExerciseRecommendation,
    HealthStatus,
    LiveSessionOutput,
    RecommendationSession,
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


def recommend_for_user(
    user: User,
    health: HealthStatus,
    sessions: List[LiveSessionOutput],
    target_region: BodyRegion,
    community: Optional[List[CommunityRecord]] = None,
    feedbacks: Optional[List[UserFeedback]] = None,
) -> List[ExerciseRecommendation]:
    """
    Main entry point for the API.

    Converts user data → rec engine inputs, applies limitation filter,
    then delegates to recommend().
    """
    history = [from_live_session(s) for s in sessions]

    # Exclude exercises whose primary region is affected by the user's limitations.
    limited_regions = {
        _JOINT_TO_REGION[j]
        for j in user.limited_joints
        if j in _JOINT_TO_REGION
    }
    exercises = [
        e for e in CATALOG
        if e.primary_region not in limited_regions
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


def from_pipeline_session(
    session,                                          # user.workout_history.WorkoutSession
    user_id: str,
    health_snapshot: Optional[Dict[BodyRegion, int]] = None,
) -> ExercisePerformanceRecord:
    """
    Convert a completed WorkoutSession (CLI pipeline) into an ExercisePerformanceRecord.
    Call AFTER history.save_session() so session.overall_score is already set.
    """
    form = round(session.overall_score / 100.0, 3)
    violations = list({
        joint
        for rep in session.rep_records
        for joint in rep.error_joints
    })
    timestamp = (
        datetime.fromisoformat(session.date)
        if isinstance(session.date, str)
        else session.date
    )
    return ExercisePerformanceRecord(
        id=str(uuid.uuid4()),
        exercise_id=session.exercise_id,
        user_id=user_id,
        session_id=session.session_id,
        timestamp=timestamp,
        difficulty_score=round(1.0 - form, 3),
        form_quality_score=form,
        completion_rate=1.0,
        violations=violations,
        reps_completed=session.total_reps,
        sets_completed=1,
        health_snapshot=health_snapshot or {r: 3 for r in BodyRegion},
    )


def load_history_from_workout_history(user_id: str, history_path: str = None) -> int:
    """
    Load all past WorkoutSessions from workout_history.json and seed the
    recommendation engine with real performance data.
    Returns the number of records loaded.
    """
    from core.user.workout_history import WorkoutHistory
    from .simulation import seed_history

    wh = WorkoutHistory(history_path)
    records = [from_pipeline_session(s, user_id) for s in wh.sessions]
    seed_history(user_id, records)
    return len(records)
