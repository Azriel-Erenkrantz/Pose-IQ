"""
User service layer.

Public functions return app_model types — the rest of the app (API, UI)
only ever touches these functions, never the internal user/ module classes.

Storage: MongoDB via core.db.get_db()
  Collection 'users'   — registered users + embedded health_ratings
  Collection 'tokens'  — auth tokens with TTL index (auto-expired by MongoDB)
  Collection 'sessions'— workout sessions (written by WorkoutHistory)

To swap storage: replace get_db() calls — everything above stays the same.
"""
from __future__ import annotations

import hashlib
import hmac
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
from core.db import get_db

TOKEN_TTL_HOURS = 24 * 7  # 1 week


# ── Internal helpers ───────────────────────────────────────────────────────────

# scrypt work factors: n must be a power of 2; 2^14 keeps login < ~100ms.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


def _hash(password: str) -> str:
    """Salted scrypt hash, stored as 'scrypt$<salt_hex>$<digest_hex>'."""
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt,
                            n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    if stored.startswith('scrypt$'):
        _, salt_hex, digest_hex = stored.split('$')
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                                n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
        return hmac.compare_digest(digest.hex(), digest_hex)
    # Legacy unsalted sha256 (pre-scrypt accounts) — accepted once, then
    # login() rewrites the stored hash with scrypt.
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy, stored)


def _doc_to_user(doc: Dict) -> User:
    return User(
        user_id             = str(doc['_id']),
        name                = doc['name'],
        email               = doc['email'],
        fitness_level       = FitnessLevel(doc.get('fitness_level', 'intermediate')),
        trainer_personality = TrainerPersonality(doc.get('trainer_personality', 'motivating')),
        target_goals        = [TargetGoal(g) for g in doc.get('target_goals', [])],
        equipment           = [Equipment(e) for e in doc.get('equipment', [])],
        limitations         = doc.get('limitations', []),
        created_at          = doc.get('created_at', datetime.now()),
    )


def _user_to_doc(user: User, password_hash: str) -> Dict:
    return {
        '_id':               user.user_id,
        'email':             user.email,
        'password_hash':     password_hash,
        'name':              user.name,
        'fitness_level':     user.fitness_level.value,
        'trainer_personality': user.trainer_personality.value,
        'target_goals':      [g.value for g in user.target_goals],
        'equipment':         [e.value for e in user.equipment],
        'limitations':       user.limitations,
        'created_at':        user.created_at,
    }


# ── Auth ───────────────────────────────────────────────────────────────────────

def _normalize_email(email: str) -> str:
    # Emails are matched exactly in Mongo — normalize once at the boundary so
    # "Demo@X.com " and "demo@x.com" are the same account.
    return email.strip().lower()


def register(req: RegisterRequest) -> Optional[AuthToken]:
    """Create a new user. Returns None if the email is already registered."""
    db = get_db()
    email = _normalize_email(req.email)
    if db.users.find_one({'email': email}):
        return None

    user = User(
        user_id             = str(uuid.uuid4()),
        name                = req.name,
        email               = email,
        fitness_level       = req.fitness_level,
        trainer_personality = req.trainer_personality,
        target_goals        = req.target_goals,
        equipment           = req.equipment,
        limitations         = req.limitations,
    )
    db.users.insert_one(_user_to_doc(user, _hash(req.password)))

    return _issue_token(db, user.user_id)


def login(req: LoginRequest) -> Optional[AuthToken]:
    """Verify credentials and return a token. Returns None on mismatch."""
    db  = get_db()
    doc = db.users.find_one({'email': _normalize_email(req.email)})
    if doc is None or not _verify_password(req.password, doc['password_hash']):
        return None
    if not doc['password_hash'].startswith('scrypt$'):
        db.users.update_one({'_id': doc['_id']},
                            {'$set': {'password_hash': _hash(req.password)}})
    return _issue_token(db, str(doc['_id']))


def verify_token(token: str) -> Optional[str]:
    """Return user_id for a valid unexpired token, else None."""
    db  = get_db()
    doc = db.tokens.find_one({'_id': token, 'expires_at': {'$gt': datetime.now()}})
    return str(doc['user_id']) if doc else None


def _issue_token(db, user_id: str) -> AuthToken:
    token      = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(hours=TOKEN_TTL_HOURS)
    db.tokens.insert_one({'_id': token, 'user_id': user_id, 'expires_at': expires_at})
    return AuthToken(user_id=user_id, token=token, expires_at=expires_at)


# ── User data ──────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> Optional[User]:
    doc = get_db().users.find_one({'_id': user_id})
    return _doc_to_user(doc) if doc else None


def update_user(user: User) -> bool:
    """Persist profile changes. Returns False if user_id not found."""
    db  = get_db()
    doc = db.users.find_one({'_id': user.user_id})
    if not doc:
        return False
    db.users.update_one({'_id': user.user_id}, {'$set': {
        'name':              user.name,
        'fitness_level':     user.fitness_level.value,
        'trainer_personality': user.trainer_personality.value,
        'target_goals':      [g.value for g in user.target_goals],
        'equipment':         [e.value for e in user.equipment],
        'limitations':       user.limitations,
    }})
    return True


# ── Health status ──────────────────────────────────────────────────────────────

def get_health_status(user_id: str) -> HealthStatus:
    """Read stored health ratings; falls back to all-5 (fully healthy) if none recorded."""
    doc = get_db().users.find_one({'_id': user_id}, {'health_ratings': 1})
    ratings_raw = (doc or {}).get('health_ratings', {})
    if ratings_raw:
        ratings = {BodyRegion(k): v for k, v in ratings_raw.items()}
    else:
        ratings = {r: 5 for r in BodyRegion}
    return HealthStatus(user_id=user_id, ratings=ratings)


def save_health_status(user_id: str, health: HealthStatus) -> bool:
    """Persist health ratings. Returns False if user not found."""
    db = get_db()
    if not db.users.find_one({'_id': user_id}):
        return False
    db.users.update_one(
        {'_id': user_id},
        {'$set': {'health_ratings': {k.value: v for k, v in health.ratings.items()}}},
    )
    return True


# ── Workout history ────────────────────────────────────────────────────────────

def get_history(user_id: str) -> List[LiveSessionOutput]:
    """Load session history for a user as app_model types."""
    try:
        from core.user.workout_history import WorkoutHistory
        return [_session_to_live(s, user_id) for s in WorkoutHistory(user_id=user_id).get_sessions()]
    except Exception:
        return []


def _session_to_live(session, user_id: str) -> LiveSessionOutput:
    reps = [
        RepResult(
            rep_number       = r.rep_number,
            form_score       = r.form_score,
            error_joints     = r.error_joints,
            duration_seconds = 0.0,
        )
        for r in session.rep_records
    ]
    date = datetime.fromisoformat(session.date) if isinstance(session.date, str) else session.date
    return LiveSessionOutput(
        session_id       = session.session_id,
        exercise_id      = session.exercise_id,
        exercise_name    = session.exercise_name,
        user_id          = user_id,
        date             = date,
        reps             = reps,
        duration_seconds = session.duration_seconds,
        overall_score    = session.overall_score,
        weight_kg        = session.weight_kg,
    )


def set_session_weight(user_id: str, session_id: str, weight_kg: Optional[float]) -> bool:
    """Record the load used in a past session. Returns False if the session
    doesn't exist or belongs to another user."""
    from core.user.workout_history import WorkoutHistory
    return WorkoutHistory(user_id=user_id).set_session_weight(session_id, weight_kg)


def save_workout_session(
    user_id: str,
    exercise_id: str,
    exercise_name: Optional[str],
    duration_seconds: float,
    weight_kg: Optional[float],
    reps: List[Dict],
) -> Optional['WorkoutSession']:  # noqa: F821 — forward ref, imported lazily
    """Persist a completed workout reported by a client (the web live-workout
    screen). Mirrors what core/pipeline.py saves at the end of a session.
    Returns the saved session (with computed overall score), or None if the
    user doesn't exist."""
    db = get_db()
    if not db.users.find_one({'_id': user_id}):
        return None

    if not exercise_name:
        doc = db.exercises.find_one({'id': exercise_id})
        exercise_name = doc['name'] if doc else exercise_id

    from core.user.workout_history import RepRecord, WorkoutHistory, new_session
    session = new_session(exercise_id, exercise_name, weight_kg=weight_kg)
    session.rep_records = [
        RepRecord(
            rep_number   = r['rep_number'],
            error_joints = list(r.get('error_joints', [])),
            form_score   = r['form_score'],
        )
        for r in reps
    ]
    session.total_reps = len(session.rep_records)
    session.duration_seconds = duration_seconds
    WorkoutHistory(user_id=user_id).save_session(session)
    return session


# ── Progress metrics ───────────────────────────────────────────────────────────

def get_progress(user_id: str) -> List[ProgressMetrics]:
    """Compute per-exercise progress metrics from session history."""
    sessions = get_history(user_id)
    if not sessions:
        return []

    by_exercise: Dict[str, List[LiveSessionOutput]] = {}
    for s in sessions:
        by_exercise.setdefault(s.exercise_id, []).append(s)

    metrics = []
    for ex_id, ex_sessions in by_exercise.items():
        sorted_sessions = sorted(ex_sessions, key=lambda s: s.date)
        recent = sorted_sessions[-3:]
        avg_score = round(sum(s.overall_score for s in recent) / len(recent), 1)

        if len(sorted_sessions) >= 2:
            diff  = sorted_sessions[-1].overall_score - sorted_sessions[0].overall_score
            trend = ScoreTrend.IMPROVING if diff > 3 else (ScoreTrend.DECLINING if diff < -3 else ScoreTrend.STABLE)
        else:
            trend = ScoreTrend.STABLE

        joint_counts: Dict[str, int] = {}
        for s in ex_sessions:
            for j in s.weak_joints:
                joint_counts[j] = joint_counts.get(j, 0) + 1

        metrics.append(ProgressMetrics(
            exercise_id      = ex_id,
            exercise_name    = ex_sessions[0].exercise_name,
            total_sessions   = len(ex_sessions),
            total_reps       = sum(s.total_reps for s in ex_sessions),
            avg_score_recent = avg_score,
            score_trend      = trend,
            weak_joints      = sorted(joint_counts.items(), key=lambda x: x[1], reverse=True),
        ))

    return metrics


# ── Onboarding options ─────────────────────────────────────────────────────────

LIMITATION_OPTIONS = [
    'right_knee', 'left_knee', 'lower_back',
    'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
]


def get_profile_setup_options():
    from core.app_model import ProfileSetupScreenData
    return ProfileSetupScreenData(
        fitness_levels        = list(FitnessLevel),
        trainer_personalities = list(TrainerPersonality),
        target_goals          = list(TargetGoal),
        equipment_options     = list(Equipment),
        limitation_options    = LIMITATION_OPTIONS,
    )
