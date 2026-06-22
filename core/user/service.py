"""
User service layer.

Public functions return app_model types — the rest of the app (API, UI)
only ever touches these functions, never the internal user/ module classes.

Storage (right now): JSON files under data/
  data/users.json          — registered users {user_id: {email, password_hash, profile}}
  data/user_profile.json   — legacy single-user CLI profile (untouched)

When the DB is connected: swap the _load_users / _save_users helpers.
Everything above them stays the same.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.app_model import (
    AuthToken,
    BodyRegion,
    Equipment,
    FitnessLevel,
    HealthStatus,
    LiveSessionOutput,
    LoginRequest,
    ProgressMetrics,
    RegisterRequest,
    RepResult,
    ScoreTrend,
    TargetGoal,
    TrainerPersonality,
    User,
)

# ── Storage path ───────────────────────────────────────────────────────────────

_USERS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'users.json')
)

TOKEN_TTL_HOURS = 24 * 7  # 1 week


# ── Internal helpers ───────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> Dict:
    if not os.path.exists(_USERS_PATH):
        return {}
    with open(_USERS_PATH) as f:
        return json.load(f)


def _save_users(users: Dict) -> None:
    os.makedirs(os.path.dirname(_USERS_PATH), exist_ok=True)
    with open(_USERS_PATH, 'w') as f:
        json.dump(users, f, indent=2, default=str)


def _dict_to_user(d: Dict) -> User:
    return User(
        user_id             = d["user_id"],
        name                = d["name"],
        email               = d["email"],
        fitness_level       = FitnessLevel(d.get("fitness_level", "intermediate")),
        trainer_personality = TrainerPersonality(d.get("trainer_personality", "motivating")),
        target_goals        = [TargetGoal(g) for g in d.get("target_goals", [])],
        equipment           = [Equipment(e) for e in d.get("equipment", [])],
        limitations         = d.get("limitations", []),
        created_at          = datetime.fromisoformat(d.get("created_at", datetime.now().isoformat())),
    )


def _user_to_dict(user: User, password_hash: str) -> Dict:
    return {
        "user_id":             user.user_id,
        "name":                user.name,
        "email":               user.email,
        "password_hash":       password_hash,
        "fitness_level":       user.fitness_level.value,
        "trainer_personality": user.trainer_personality.value,
        "target_goals":        [g.value for g in user.target_goals],
        "equipment":           [e.value for e in user.equipment],
        "limitations":         user.limitations,
        "created_at":          user.created_at.isoformat(),
    }


# ── Auth ───────────────────────────────────────────────────────────────────────

def register(req: RegisterRequest) -> Optional[AuthToken]:
    """
    Create a new user account.
    Returns None if the email is already registered.
    """
    users = _load_users()

    if any(u["email"] == req.email for u in users.values()):
        return None  # email already taken

    user = User(
        user_id             = str(uuid.uuid4()),
        name                = req.name,
        email               = req.email,
        fitness_level       = req.fitness_level,
        trainer_personality = req.trainer_personality,
        target_goals        = req.target_goals,
        equipment           = req.equipment,
        limitations         = req.limitations,
    )
    users[user.user_id] = _user_to_dict(user, _hash(req.password))
    _save_users(users)

    return AuthToken(
        user_id    = user.user_id,
        token      = str(uuid.uuid4()),
        expires_at = datetime.now() + timedelta(hours=TOKEN_TTL_HOURS),
    )


def login(req: LoginRequest) -> Optional[AuthToken]:
    """
    Verify credentials and return a token.
    Returns None if email not found or password is wrong.
    """
    users = _load_users()
    match = next(
        (u for u in users.values() if u["email"] == req.email),
        None,
    )
    if match is None or match["password_hash"] != _hash(req.password):
        return None

    return AuthToken(
        user_id    = match["user_id"],
        token      = str(uuid.uuid4()),
        expires_at = datetime.now() + timedelta(hours=TOKEN_TTL_HOURS),
    )


# ── User data ──────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> Optional[User]:
    users = _load_users()
    raw = users.get(user_id)
    return _dict_to_user(raw) if raw else None


def update_user(user: User) -> bool:
    """Persist profile changes. Returns False if user_id not found."""
    users = _load_users()
    if user.user_id not in users:
        return False
    existing = users[user.user_id]
    users[user.user_id] = _user_to_dict(user, existing["password_hash"])
    _save_users(users)
    return True


# ── Health status ──────────────────────────────────────────────────────────────

def get_health_status(user_id: str) -> HealthStatus:
    """
    Reads last recorded health status.
    Falls back to all-5 (fully healthy) when no record exists.
    Health status recording is a future feature — placeholder for now.
    """
    return HealthStatus(
        user_id  = user_id,
        ratings  = {r: 5 for r in BodyRegion},
        recorded_at = datetime.now(),
    )


# ── Workout history ────────────────────────────────────────────────────────────

def get_history(user_id: str) -> List[LiveSessionOutput]:
    """
    Load session history for a user, returned as app_model types.
    Reads from data/workout_history.json via the existing WorkoutHistory class.
    Returns empty list if no history exists.
    """
    try:
        from core.user.workout_history import WorkoutHistory
        wh = WorkoutHistory(user_id=user_id)
        sessions = wh.get_sessions()
        return [_workout_session_to_live(s, user_id) for s in sessions]
    except Exception:
        return []


def _workout_session_to_live(session, user_id: str) -> LiveSessionOutput:
    reps = [
        RepResult(
            rep_number       = r.rep_number,
            form_score       = r.form_score,
            error_joints     = r.error_joints,
            duration_seconds = 0.0,
        )
        for r in session.rep_records
    ]
    return LiveSessionOutput(
        session_id       = session.session_id,
        exercise_id      = session.exercise_id,
        exercise_name    = session.exercise_name,
        user_id          = user_id,
        date             = datetime.fromisoformat(session.date) if isinstance(session.date, str) else session.date,
        reps             = reps,
        duration_seconds = session.duration_seconds,
        overall_score    = session.overall_score,
    )


# ── Progress metrics ───────────────────────────────────────────────────────────

def get_progress(user_id: str) -> List[ProgressMetrics]:
    """
    Compute per-exercise progress metrics from session history.
    Returns empty list if no history exists.
    """
    sessions = get_history(user_id)
    if not sessions:
        return []

    by_exercise: Dict[str, List[LiveSessionOutput]] = {}
    for s in sessions:
        by_exercise.setdefault(s.exercise_id, []).append(s)

    metrics = []
    for ex_id, ex_sessions in by_exercise.items():
        ex_sessions_sorted = sorted(ex_sessions, key=lambda s: s.date)
        recent = ex_sessions_sorted[-3:]

        avg_score = round(sum(s.overall_score for s in recent) / len(recent), 1)

        if len(ex_sessions_sorted) >= 2:
            first_score = ex_sessions_sorted[0].overall_score
            last_score  = ex_sessions_sorted[-1].overall_score
            diff = last_score - first_score
            trend = ScoreTrend.IMPROVING if diff > 3 else (ScoreTrend.DECLINING if diff < -3 else ScoreTrend.STABLE)
        else:
            trend = ScoreTrend.STABLE

        joint_counts: Dict[str, int] = {}
        for s in ex_sessions:
            for j in s.weak_joints:
                joint_counts[j] = joint_counts.get(j, 0) + 1

        weak_joints = sorted(joint_counts.items(), key=lambda x: x[1], reverse=True)

        metrics.append(ProgressMetrics(
            exercise_id      = ex_id,
            exercise_name    = ex_sessions[0].exercise_name,
            total_sessions   = len(ex_sessions),
            total_reps       = sum(s.total_reps for s in ex_sessions),
            avg_score_recent = avg_score,
            score_trend      = trend,
            weak_joints      = weak_joints,
        ))

    return metrics


# ── Onboarding options ─────────────────────────────────────────────────────────

LIMITATION_OPTIONS = [
    "right_knee",
    "left_knee",
    "lower_back",
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
]


def get_profile_setup_options():
    """Returns the static options needed to render the onboarding screen."""
    from core.app_model import ProfileSetupScreenData
    return ProfileSetupScreenData(
        fitness_levels        = list(FitnessLevel),
        trainer_personalities = list(TrainerPersonality),
        target_goals          = list(TargetGoal),
        equipment_options     = list(Equipment),
        limitation_options    = LIMITATION_OPTIONS,
    )
