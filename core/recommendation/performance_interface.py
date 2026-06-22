"""
Contract and stub for the performance-monitoring ML model.

Defines:
  - ExercisePerformanceRecord  (in models.py) — the data the real model must return
  - simulate_performance()     — deterministic-override-friendly stub used in tests / demo

To convert a completed pipeline WorkoutSession into an ExercisePerformanceRecord,
see recommendation/bridge.py (optional integration adapter, not imported here).
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .models import BodyRegion, ExercisePerformanceRecord


def simulate_performance(
    user_id: str,
    session_id: str,
    exercise_id: str,
    health_snapshot: Dict[BodyRegion, int],
    *,
    override_difficulty: Optional[float] = None,
    override_form: Optional[float] = None,
    override_violations: Optional[List[str]] = None,
) -> ExercisePerformanceRecord:
    """
    Stand-in for the performance-monitoring ML model.

    In production, replace this with the real model call:
        return performance_model.predict(session_video, exercise_id, user_id)

    The override_* kwargs inject deterministic values for tests and demo scripts.
    """
    difficulty = override_difficulty if override_difficulty is not None \
        else round(random.uniform(0.20, 0.90), 2)

    form = override_form if override_form is not None \
        else round(random.uniform(0.50, 1.00), 2)

    violations = override_violations if override_violations is not None \
        else random.choice([
            [],
            ["minor_form_break"],
            ["joint_misalignment"],
            ["incomplete_range_of_motion", "speed_too_fast"],
        ])

    reps = max(1, int((1.0 - difficulty) * 12) + 3)
    sets = random.choice([2, 3])

    return ExercisePerformanceRecord(
        id=str(uuid.uuid4()),
        exercise_id=exercise_id,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now(),
        difficulty_score=difficulty,
        form_quality_score=form,
        completion_rate=round(max(0.1, 1.0 - difficulty * 0.3), 2),
        violations=violations,
        reps_completed=reps,
        sets_completed=sets,
        health_snapshot=health_snapshot,
    )
