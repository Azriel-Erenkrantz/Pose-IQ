"""
Pose-IQ REST API — user module.

Run from the project root:
    flask --app api.app run --debug

All responses are JSON. Auth endpoints return an AuthToken.
Protected endpoints expect:  Authorization: Bearer <token>

Dev endpoints (/api/dev/*) bypass auth and return fake data —
use them while building the mobile UI before a real user exists.

When MongoDB replaces JSON files: only service.py changes.
Routes here don't touch storage — they call service.py only.
"""
from __future__ import annotations

import dataclasses
import os
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from core.app_model import (
    BodyRegion,
    DashboardData,
    HealthStatus,
    HistoryScreenData,
    LoginRequest,
    RegisterRequest,
    Equipment,
    FitnessLevel,
    TargetGoal,
    TrainerPersonality,
)
from core.user import service
from core.user import fake_data
from core.recommendation.bridge import recommend_for_user
from core.recommendation.overload import recommend_weights_for_user

app = Flask(__name__)

# FRONTEND_ORIGIN: comma-separated allowed origins for the deployed frontend
# (e.g. "https://pose-iq.vercel.app"). Defaults to "*" so local dev and the
# desktop pipeline work with zero setup — tighten this in production via env.
_origins = os.environ.get('FRONTEND_ORIGIN', '*')
CORS(app, origins=_origins.split(',') if _origins != '*' else '*')


# ── Serialization ──────────────────────────────────────────────────────────────

def _serialize(obj: Any) -> Any:
    """Recursively convert app_model dataclasses to JSON-safe dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in vars(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_serialize(i) for i in obj]
    return obj


def ok(data: Any, status: int = 200):
    return jsonify(_serialize(data)), status


def err(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ── Auth guard ────────────────────────────────────────────────────────────────

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return err("Missing or invalid Authorization header", 401)
        token = auth[len("Bearer "):]
        token_user_id = service.verify_token(token)
        if token_user_id is None:
            return err("Token invalid or expired", 401)
        # A valid token only grants access to its own user's resources.
        route_user_id = kwargs.get("user_id")
        if route_user_id is not None and route_user_id != token_user_id:
            return err("Forbidden", 403)
        return f(*args, **kwargs)
    return wrapper


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register():
    body = request.get_json(silent=True) or {}
    try:
        req = RegisterRequest(
            name                = body["name"],
            email               = body["email"],
            password            = body["password"],
            fitness_level       = FitnessLevel(body.get("fitness_level", "intermediate")),
            trainer_personality = TrainerPersonality(body.get("trainer_personality", "motivating")),
            target_goals        = [TargetGoal(g) for g in body.get("target_goals", [])],
            equipment           = [Equipment(e) for e in body.get("equipment", [])],
            limitations         = body.get("limitations", []),
        )
    except (KeyError, ValueError) as e:
        return err(f"Invalid request: {e}")

    token = service.register(req)
    if token is None:
        return err("Email already registered", 409)
    return ok(token, 201)


@app.post("/api/auth/login")
def login():
    body = request.get_json(silent=True) or {}
    try:
        req = LoginRequest(email=body["email"], password=body["password"])
    except KeyError as e:
        return err(f"Missing field: {e}")

    token = service.login(req)
    if token is None:
        return err("Invalid email or password", 401)
    return ok(token)


@app.get("/api/auth/options")
def onboarding_options():
    """All picker options the onboarding screen needs — no auth required."""
    return ok(service.get_profile_setup_options())


# ── User routes ────────────────────────────────────────────────────────────────

@app.get("/api/user/<user_id>")
@require_auth
def get_user(user_id: str):
    user = service.get_user(user_id)
    if user is None:
        return err("User not found", 404)
    return ok(user)


@app.put("/api/user/<user_id>")
@require_auth
def update_user(user_id: str):
    user = service.get_user(user_id)
    if user is None:
        return err("User not found", 404)

    body = request.get_json(silent=True) or {}
    try:
        if "fitness_level" in body:
            user.fitness_level = FitnessLevel(body["fitness_level"])
        if "trainer_personality" in body:
            user.trainer_personality = TrainerPersonality(body["trainer_personality"])
        if "target_goals" in body:
            user.target_goals = [TargetGoal(g) for g in body["target_goals"]]
        if "equipment" in body:
            user.equipment = [Equipment(e) for e in body["equipment"]]
        if "limitations" in body:
            user.limitations = body["limitations"]
        if "name" in body:
            user.name = body["name"]
    except ValueError as e:
        return err(f"Invalid value: {e}")

    service.update_user(user)
    return ok(user)


@app.get("/api/user/<user_id>/health")
@require_auth
def get_health(user_id: str):
    return ok(service.get_health_status(user_id))


@app.put("/api/user/<user_id>/health")
@require_auth
def update_health(user_id: str):
    body = request.get_json(silent=True) or {}
    try:
        ratings = {BodyRegion(k): int(v) for k, v in body.items()}
    except (ValueError, TypeError) as e:
        return err(f"Invalid ratings: {e}")
    if not all(1 <= v <= 5 for v in ratings.values()):
        return err("Each rating must be between 1 and 5")
    health = HealthStatus(user_id=user_id, ratings=ratings)
    if not service.save_health_status(user_id, health):
        return err("User not found", 404)
    return ok(health)


@app.get("/api/exercises")
def list_exercises():
    """Full exercise model for clients that run the state machine locally
    (the web live-workout screen): phases with measured angle ranges +
    corrections, global constraints, mandatory start joints.

    `ready` is False until the trainer has written angle ranges to Mongo —
    the state machine can't track phases without them."""
    from core.exercise.exercise_model import ExerciseModel
    try:
        from core.db import get_db
        model = ExerciseModel.from_mongo(get_db())
        if not model.list_exercises():
            model = ExerciseModel()
    except Exception:
        model = ExerciseModel()

    out = []
    for ex_id in model.list_exercises():
        ex = model.get_exercise(ex_id)
        data = _serialize(ex)
        data["ready"] = bool(ex.phases) and all(ph.angles for ph in ex.phases)
        out.append(data)
    return ok(out)


@app.get("/api/user/<user_id>/history")
@require_auth
def get_history(user_id: str):
    return ok(service.get_history(user_id))


@app.post("/api/user/<user_id>/sessions")
@require_auth
def create_session(user_id: str):
    """Save a workout completed in a client (web live-workout screen)."""
    body = request.get_json(silent=True) or {}
    exercise_id = body.get("exercise_id")
    reps = body.get("reps")
    if not exercise_id or not isinstance(reps, list) or not reps:
        return err("exercise_id and a non-empty reps list are required")

    weight = body.get("weight_kg")
    if weight is not None:
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            return err("weight_kg must be a number or null")
        if not 0 <= weight <= 300:
            return err("weight_kg must be between 0 and 300")

    try:
        duration = float(body.get("duration_seconds", 0) or 0)
        rep_records = [
            {
                "rep_number":   int(r["rep_number"]),
                "error_joints": list(r.get("error_joints", [])),
                "form_score":   float(r["form_score"]),
            }
            for r in reps
        ]
    except (KeyError, TypeError, ValueError) as e:
        return err(f"Invalid reps payload: {e}")

    session = service.save_workout_session(
        user_id, exercise_id, body.get("exercise_name"),
        duration, weight, rep_records,
    )
    if session is None:
        return err("User not found", 404)
    return ok({
        "session_id":    session.session_id,
        "total_reps":    session.total_reps,
        "overall_score": session.overall_score,
        "weight_kg":     session.weight_kg,
    }, 201)


@app.get("/api/user/<user_id>/progress")
@require_auth
def get_progress(user_id: str):
    return ok(service.get_progress(user_id))


@app.put("/api/user/<user_id>/sessions/<session_id>/weight")
@require_auth
def set_session_weight(user_id: str, session_id: str):
    body = request.get_json(silent=True) or {}
    if "weight_kg" not in body:
        return err("Missing field: weight_kg")
    weight = body["weight_kg"]
    if weight is not None:
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            return err("weight_kg must be a number or null")
        if not 0 <= weight <= 300:
            return err("weight_kg must be between 0 and 300")
    if not service.set_session_weight(user_id, session_id, weight):
        return err("Session not found", 404)
    return ok({"session_id": session_id, "weight_kg": weight})


@app.get("/api/user/<user_id>/weights")
@require_auth
def get_weight_recommendations(user_id: str):
    user = service.get_user(user_id)
    if user is None:
        return err("User not found", 404)
    return ok(recommend_weights_for_user(user, service.get_history(user_id)))


@app.get("/api/dashboard/<user_id>")
@require_auth
def get_dashboard(user_id: str):
    user = service.get_user(user_id)
    if user is None:
        return err("User not found", 404)

    health   = service.get_health_status(user_id)
    sessions = service.get_history(user_id)
    recs     = recommend_for_user(user, health, sessions)

    dashboard = DashboardData(
        user             = user,
        health_status    = health,
        recommendations  = recs,
        recent_sessions  = sessions[:5],
        progress_summary = service.get_progress(user_id),
        injury_risk      = None,
        weight_recommendations = recommend_weights_for_user(user, sessions),
    )
    return ok(dashboard)


# ── Dev / fake-data routes (no auth, for mobile UI development) ───────────────

@app.get("/api/dev/user")
def dev_user():
    return ok(fake_data.fake_user())


@app.get("/api/dev/history")
def dev_history():
    return ok(fake_data.fake_sessions())


@app.get("/api/dev/progress")
def dev_progress():
    return ok(fake_data.fake_progress_metrics())


@app.get("/api/dev/dashboard")
def dev_dashboard():
    user     = fake_data.fake_user()
    health   = fake_data.fake_health_status()
    sessions = fake_data.fake_sessions()
    progress = fake_data.fake_progress_metrics()
    risk     = fake_data.fake_injury_risk()

    dashboard = DashboardData(
        user             = user,
        health_status    = health,
        recommendations  = [],
        recent_sessions  = sessions[:5],
        progress_summary = progress,
        injury_risk      = risk,
    )
    return ok(dashboard)


@app.get("/api/dev/options")
def dev_options():
    return ok(service.get_profile_setup_options())


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/api/ping")
def ping():
    return ok({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
