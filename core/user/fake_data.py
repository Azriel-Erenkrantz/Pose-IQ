"""
Fake data for the user module.

Used during UI development and testing — no real user, no files needed.
Every function returns a valid app_model instance so the mobile app
and API layer can be built and tested without a real user or DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from core.app_model import (
    AuthToken,
    BodyRegion,
    Equipment,
    FitnessLevel,
    HealthStatus,
    InjuryRisk,
    LiveSessionOutput,
    ProgressMetrics,
    RepResult,
    ScoreTrend,
    TargetGoal,
    TrainerPersonality,
    User,
)

# ── Fake user ──────────────────────────────────────────────────────────────────

FAKE_USER_ID    = "fake-user-001"
FAKE_USER_EMAIL = "demo@poseiq.com"
FAKE_USER_TOKEN = "fake-token-abc123"


def fake_user() -> User:
    return User(
        user_id             = FAKE_USER_ID,
        name                = "Demo User",
        email               = FAKE_USER_EMAIL,
        fitness_level       = FitnessLevel.INTERMEDIATE,
        trainer_personality = TrainerPersonality.MOTIVATING,
        target_goals        = [TargetGoal.LEGS, TargetGoal.ABS],
        equipment           = [Equipment.DUMBBELLS],
        limitations         = [],
        created_at          = datetime(2025, 1, 1),
    )


def fake_auth_token() -> AuthToken:
    return AuthToken(
        user_id    = FAKE_USER_ID,
        token      = FAKE_USER_TOKEN,
        expires_at = datetime(2099, 12, 31),
    )


# ── Fake health ────────────────────────────────────────────────────────────────

def fake_health_status() -> HealthStatus:
    return HealthStatus(
        user_id  = FAKE_USER_ID,
        ratings  = {
            BodyRegion.UPPER: 5,
            BodyRegion.CORE:  4,
            BodyRegion.LOWER: 3,
        },
        recorded_at = datetime.now(),
    )


# ── Fake session history ───────────────────────────────────────────────────────

def _make_session(
    session_id:    str,
    exercise_id:   str,
    exercise_name: str,
    days_ago:      int,
    scores:        List[float],
    error_joints:  List[str],
) -> LiveSessionOutput:
    reps = [
        RepResult(
            rep_number       = i + 1,
            form_score       = scores[i % len(scores)],
            error_joints     = error_joints if scores[i % len(scores)] < 75 else [],
            duration_seconds = 3.5,
        )
        for i in range(len(scores))
    ]
    overall = round(sum(r.form_score for r in reps) / len(reps), 1)
    return LiveSessionOutput(
        session_id       = session_id,
        exercise_id      = exercise_id,
        exercise_name    = exercise_name,
        user_id          = FAKE_USER_ID,
        date             = datetime.now() - timedelta(days=days_ago),
        reps             = reps,
        duration_seconds = len(scores) * 3.5,
        overall_score    = overall,
    )


def fake_sessions() -> List[LiveSessionOutput]:
    return [
        _make_session("s1", "squat",       "Squat",        days_ago=1,  scores=[72, 75, 78, 80, 82], error_joints=["spine", "right_knee"]),
        _make_session("s2", "squat",       "Squat",        days_ago=4,  scores=[65, 68, 70, 72, 74], error_joints=["spine", "right_knee"]),
        _make_session("s3", "squat",       "Squat",        days_ago=8,  scores=[60, 62, 65, 68, 70], error_joints=["spine"]),
        _make_session("s4", "plank",       "Plank",        days_ago=2,  scores=[88, 85, 90, 87, 92], error_joints=["hips"]),
        _make_session("s5", "plank",       "Plank",        days_ago=6,  scores=[80, 82, 85, 88, 86], error_joints=["hips"]),
        _make_session("s6", "bicep_curl",  "Bicep Curl",   days_ago=3,  scores=[91, 93, 90, 95, 92], error_joints=[]),
        _make_session("s7", "lunge",       "Lunge",        days_ago=5,  scores=[70, 73, 75, 78, 80], error_joints=["left_knee"]),
        _make_session("s8", "shoulder_press", "Shoulder Press", days_ago=7, scores=[83, 85, 88, 84, 87], error_joints=[]),
    ]


# ── Fake progress metrics ──────────────────────────────────────────────────────

def fake_progress_metrics() -> List[ProgressMetrics]:
    return [
        ProgressMetrics(
            exercise_id      = "squat",
            exercise_name    = "Squat",
            total_sessions   = 3,
            total_reps       = 15,
            avg_score_recent = 77.4,
            score_trend      = ScoreTrend.IMPROVING,
            weak_joints      = [("spine", 8), ("right_knee", 5)],
        ),
        ProgressMetrics(
            exercise_id      = "plank",
            exercise_name    = "Plank",
            total_sessions   = 2,
            total_reps       = 10,
            avg_score_recent = 87.2,
            score_trend      = ScoreTrend.IMPROVING,
            weak_joints      = [("hips", 3)],
        ),
        ProgressMetrics(
            exercise_id      = "bicep_curl",
            exercise_name    = "Bicep Curl",
            total_sessions   = 1,
            total_reps       = 5,
            avg_score_recent = 92.2,
            score_trend      = ScoreTrend.STABLE,
            weak_joints      = [],
        ),
        ProgressMetrics(
            exercise_id      = "lunge",
            exercise_name    = "Lunge",
            total_sessions   = 1,
            total_reps       = 5,
            avg_score_recent = 75.2,
            score_trend      = ScoreTrend.STABLE,
            weak_joints      = [("left_knee", 2)],
        ),
    ]


# ── Fake injury risk ───────────────────────────────────────────────────────────

def fake_injury_risk() -> InjuryRisk:
    return InjuryRisk(
        user_id               = FAKE_USER_ID,
        risk_areas            = ["right_knee"],
        compensation_patterns = ["slight hip shift right during squat"],
        asymmetry_score       = 0.2,
        overall_risk          = 0.3,
        recommendation        = "Focus on hip alignment during squats. Consider adding single-leg exercises to balance strength.",
    )
