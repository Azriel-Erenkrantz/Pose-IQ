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

app = Flask(__name__)
CORS(app)


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


@app.get("/api/user/<user_id>/history")
@require_auth
def get_history(user_id: str):
    return ok(service.get_history(user_id))


@app.get("/api/user/<user_id>/progress")
@require_auth
def get_progress(user_id: str):
    return ok(service.get_progress(user_id))


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
