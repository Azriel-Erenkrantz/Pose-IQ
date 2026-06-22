"""
Optional integration adapter: connects core/pipeline.py to the recommendation engine.

NOT imported by the recommendation module itself — import this explicitly when
wiring up the pipeline:

    from recommendation.bridge import from_pipeline_session, load_history_from_workout_history

How to use in pipeline.py
--------------------------
1. At __init__ time (after loading the user profile and workout history):
       from recommendation.bridge import load_history_from_workout_history
       load_history_from_workout_history(self.profile.ensure_user_id())

2. At session end (after history.save_session, which triggers calculate_score):
       from recommendation.bridge import from_pipeline_session
       from recommendation.simulation import seed_history
       seed_history(user_id, [from_pipeline_session(self.session, user_id)])
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Optional

from .models import BodyRegion, ExercisePerformanceRecord


def from_pipeline_session(
    session,                                          # user.workout_history.WorkoutSession
    user_id: str,
    health_snapshot: Optional[Dict[BodyRegion, int]] = None,
) -> ExercisePerformanceRecord:
    """
    Convert a completed WorkoutSession into an ExercisePerformanceRecord.

    Call AFTER history.save_session() so session.overall_score is already set.

    Field mapping:
        form_quality_score  = session.overall_score / 100
        difficulty_score    = 1 − form_quality_score
        violations          = unique error joints across all reps
        reps_completed      = session.total_reps
        sets_completed      = 1   (pipeline = one continuous set per session)
        completion_rate     = 1.0 (user completed the session)
        health_snapshot     = caller-supplied, or neutral 3/5 for all regions
    """
    form = round(session.overall_score / 100.0, 3)
    difficulty = round(1.0 - form, 3)

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

    snapshot = health_snapshot or {r: 3 for r in BodyRegion}

    return ExercisePerformanceRecord(
        id=str(uuid.uuid4()),
        exercise_id=session.exercise_id,
        user_id=user_id,
        session_id=session.session_id,
        timestamp=timestamp,
        difficulty_score=difficulty,
        form_quality_score=form,
        completion_rate=1.0,
        violations=violations,
        reps_completed=session.total_reps,
        sets_completed=1,
        health_snapshot=snapshot,
    )


def load_history_from_workout_history(user_id: str, history_path: str = None) -> int:
    """
    Load all past WorkoutSessions from workout_history.json and seed the
    recommendation engine with real performance data.

    Returns the number of records loaded.
    Imports WorkoutHistory lazily so this file can be imported from outside core/.
    """
    from user.workout_history import WorkoutHistory
    from .simulation import seed_history

    wh = WorkoutHistory(history_path)
    records = [from_pipeline_session(s, user_id) for s in wh.sessions]
    seed_history(user_id, records)
    return len(records)
