"""
UI simulation layer.

All public functions mimic a button press or form submission in the (not yet
built) UI.  Each one mutates a UserSession and re-runs the recommender so the
caller always sees an up-to-date recommendation list.

In-memory stores (_sessions, _history, _feedbacks) are intentionally shaped as
plain lists of dataclasses so swapping them for DB queries later requires only
changing the store/fetch calls in _save_* / _load_* helpers.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .catalog import CATALOG, COMMUNITY_DATA
from .models import (
    BodyRegion,
    ExercisePerformanceRecord,
    HealthStatus,
    UserFeedback,
    UserSession,
)
from .performance_interface import simulate_performance
from .recommender import recommend

# ── In-memory stores (replace with DB calls in production) ─────────────────
_sessions:  Dict[str, UserSession]            = {}
_history:   List[ExercisePerformanceRecord]   = []
_feedbacks: List[UserFeedback]                = []


# ── Session lifecycle ───────────────────────────────────────────────────────

def start_session(
    user_id: str,
    target_region: BodyRegion,
    health_ratings: Dict[BodyRegion, int],
) -> UserSession:
    """Simulates: user selects target region and submits their health ratings."""
    health = HealthStatus(user_id=user_id, ratings=health_ratings)
    session = UserSession(
        id=str(uuid.uuid4()),
        user_id=user_id,
        target_region=target_region,
        health_status=health,
    )
    _sessions[session.id] = session
    _refresh(session)
    return session


def change_target_region(session_id: str, new_region: BodyRegion) -> UserSession:
    """Simulates: user presses the 'Change Body Region' button."""
    session = _sessions[session_id]
    session.target_region = new_region
    _refresh(session)
    return session


def update_health_status(session_id: str, region: BodyRegion, rating: int) -> UserSession:
    """Simulates: user moves a health-rating slider (1–5) for one body region."""
    session = _sessions[session_id]
    session.health_status.ratings[region] = rating
    session.health_status.recorded_at = datetime.now()
    _refresh(session)
    return session


# ── Exercise flow ───────────────────────────────────────────────────────────

def complete_exercise(
    session_id: str,
    exercise_id: str,
    *,
    difficulty: Optional[float] = None,
    form_quality: Optional[float] = None,
    violations: Optional[List[str]] = None,
) -> ExercisePerformanceRecord:
    """
    Simulates: performance-monitoring model returns results after the user
    finishes an exercise set.

    Pass difficulty / form_quality / violations to inject deterministic data
    (used by demo scripts); omit them for random simulation.
    """
    session = _sessions[session_id]
    record = simulate_performance(
        user_id=session.user_id,
        session_id=session_id,
        exercise_id=exercise_id,
        health_snapshot=dict(session.health_status.ratings),
        override_difficulty=difficulty,
        override_form=form_quality,
        override_violations=violations,
    )
    session.completed_exercises.append(record)
    _history.append(record)
    _refresh(session)
    return record


def submit_feedback(
    session_id: str,
    exercise_id: str,
    rating: int,
    comment: Optional[str] = None,
) -> UserFeedback:
    """Simulates: user submits 1–5 star post-exercise feedback."""
    session = _sessions[session_id]
    feedback = UserFeedback(
        id=str(uuid.uuid4()),
        exercise_id=exercise_id,
        user_id=session.user_id,
        session_id=session_id,
        timestamp=datetime.now(),
        rating=rating,
        comment=comment,
    )
    session.feedbacks.append(feedback)
    _feedbacks.append(feedback)
    _refresh(session)
    return feedback


def get_session(session_id: str) -> UserSession:
    return _sessions[session_id]


# ── Seed helper (for demo / testing) ───────────────────────────────────────

def seed_history(
    user_id: str,
    records: List[ExercisePerformanceRecord],
) -> None:
    """Injects pre-existing performance history for a user (bypasses session)."""
    _history.extend(records)




# ── Internal ────────────────────────────────────────────────────────────────

def _refresh(session: UserSession) -> None:
    user_history   = [r for r in _history   if r.user_id == session.user_id]
    user_feedbacks = [f for f in _feedbacks if f.user_id == session.user_id]
    session.recommendations = recommend(
        exercises=CATALOG,
        target_region=session.target_region,
        health=session.health_status,
        history=user_history,
        community=COMMUNITY_DATA,
        feedbacks=user_feedbacks,
    )


# ── Display helper ──────────────────────────────────────────────────────────

def print_recommendations(session: UserSession, top_n: int = 5) -> None:
    scenario = session.recommendations[0].scenario.value if session.recommendations else "N/A"
    health_str = "  ".join(
        f"{r.value}: {session.health_status.get(r)}/5" for r in BodyRegion
    )
    print(f"\n{'─'*62}")
    print(f"  Region : {session.target_region.value.upper()}")
    print(f"  Health : {health_str}")
    print(f"  Scenario: {scenario}")
    print(f"{'─'*62}")
    for i, rec in enumerate(session.recommendations[:top_n], 1):
        bar = "█" * int(rec.score * 10) + "░" * (10 - int(rec.score * 10))
        print(f"  {i}. {rec.exercise.name:<18} [{bar}] {rec.score:.2f}")
        print(f"     {rec.reason}")
        signals = []
        if rec.community_score is not None:
            signals.append(f"community={rec.community_score:.2f}")
        if rec.feedback_score is not None:
            signals.append(f"feedback={rec.feedback_score:.2f}")
        if signals:
            print(f"     signals: {', '.join(signals)}")
    print()
