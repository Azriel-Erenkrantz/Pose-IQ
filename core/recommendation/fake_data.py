"""
Deterministic fake scenarios for testing and demo.
No external dependencies — fully self-contained.

Each scenario() call returns a FakeScenario with:
  - a fixed user_id
  - a HealthStatus with explicit ratings
  - a target BodyRegion
  - a List[ExercisePerformanceRecord] with exact, predictable values
  - a description stating what the expected recommendation ordering should be

Using exact values (not random.uniform) means every test assertion
about "exercise A scores higher than B" will hold on every run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

from .models import BodyRegion, ExercisePerformanceRecord, HealthStatus, UserFeedback


@dataclass
class FakeScenario:
    user_id: str
    health: HealthStatus
    target_region: BodyRegion
    history: List[ExercisePerformanceRecord]
    feedbacks: List[UserFeedback]
    description: str


# ── Internal builder ──────────────────────────────────────────────────────────

def _rec(
    user_id: str,
    exercise_id: str,
    difficulty: float,
    form: float,
    violations: List[str],
    health: Dict[BodyRegion, int],
    days_ago: int = 7,
) -> ExercisePerformanceRecord:
    return ExercisePerformanceRecord(
        id=str(uuid.uuid4()),
        exercise_id=exercise_id,
        user_id=user_id,
        session_id=f"seed-{exercise_id}",
        timestamp=datetime.now() - timedelta(days=days_ago),
        difficulty_score=difficulty,
        form_quality_score=form,
        completion_rate=round(max(0.1, 1.0 - difficulty * 0.3), 2),
        violations=violations,
        reps_completed=max(1, int((1.0 - difficulty) * 12) + 3),
        sets_completed=3,
        health_snapshot=health,
    )


# ── Scenarios ─────────────────────────────────────────────────────────────────

def scenario_all_healthy() -> FakeScenario:
    """
    ALL_HEALTHY — user fully healthy, training upper body.

    shoulder_press: avg_difficulty ≈ 0.815  →  personal_score = 0.815 (HIGH)
    push_up:        avg_difficulty ≈ 0.440  →  personal_score = 0.440
    bicep_curl:     avg_difficulty ≈ 0.290  →  personal_score = 0.290 (LOW)

    Expected order: shoulder_press > push_up > bicep_curl
    Rationale: at full health, tackle the exercises that were historically hard.
    """
    uid = "user_healthy"
    health = {BodyRegion.UPPER: 5, BodyRegion.CORE: 5, BodyRegion.LOWER: 5}
    return FakeScenario(
        user_id=uid,
        health=HealthStatus(uid, health),
        target_region=BodyRegion.UPPER,
        history=[
            _rec(uid, "shoulder_press", 0.82, 0.55, ["incomplete_range_of_motion"], health, 14),
            _rec(uid, "shoulder_press", 0.81, 0.57, ["speed_too_fast"], health, 7),
            _rec(uid, "push_up",        0.46, 0.80, ["minor_form_break"], health, 10),
            _rec(uid, "push_up",        0.42, 0.83, [], health, 4),
            _rec(uid, "bicep_curl",     0.28, 0.92, [], health, 12),
            _rec(uid, "bicep_curl",     0.30, 0.90, [], health, 5),
        ],
        feedbacks=[],
        description="ALL_HEALTHY upper: shoulder_press (historically hardest) should rank first",
    )


def scenario_core_pain() -> FakeScenario:
    """
    TARGET_HEALTHY_ADJACENT_NOT — upper healthy, core hurts (core is adjacent to upper).

    push_up:        avg_difficulty = 0.440  →  score peaks near 0.45, high
    bicep_curl:     avg_difficulty = 0.280  →  moderate
    shoulder_press: avg_difficulty = 0.815  →  penalised (>0.75 threshold)

    Expected order: push_up ≈ bicep_curl >> shoulder_press
    Rationale: with core issues, avoid exercises that overload it.
    """
    uid = "user_core_pain"
    health = {BodyRegion.UPPER: 5, BodyRegion.CORE: 2, BodyRegion.LOWER: 5}
    return FakeScenario(
        user_id=uid,
        health=HealthStatus(uid, health),
        target_region=BodyRegion.UPPER,
        history=[
            _rec(uid, "shoulder_press", 0.82, 0.55, ["incomplete_range_of_motion"], health, 10),
            _rec(uid, "push_up",        0.43, 0.86, [], health, 8),
            _rec(uid, "push_up",        0.45, 0.84, [], health, 3),
            _rec(uid, "bicep_curl",     0.27, 0.93, [], health, 6),
            _rec(uid, "bicep_curl",     0.29, 0.91, [], health, 2),
        ],
        feedbacks=[],
        description="TARGET_HEALTHY_ADJACENT_NOT: push_up (moderate) >> shoulder_press (too hard)",
    )


def scenario_upper_injured() -> FakeScenario:
    """
    TARGET_UNHEALTHY — upper body injury (2/5).

    bicep_curl:  difficulty=0.92, form=0.45  →  near-zero (diff≥0.90 on injured region)
    push_up:     difficulty=0.30, form=0.90  →  good (easy + great form)
    tricep_dip:  no history                  →  0.30 (no history, unsafe default)

    Expected: push_up >> tricep_dip >> bicep_curl
    """
    uid = "user_upper_injured"
    health = {BodyRegion.UPPER: 2, BodyRegion.CORE: 4, BodyRegion.LOWER: 5}
    return FakeScenario(
        user_id=uid,
        health=HealthStatus(uid, health),
        target_region=BodyRegion.UPPER,
        history=[
            _rec(uid, "bicep_curl", 0.92, 0.45, ["joint_misalignment"], health, 5),
            _rec(uid, "push_up",    0.30, 0.90, [], health, 7),
            _rec(uid, "push_up",    0.28, 0.92, [], health, 3),
        ],
        feedbacks=[],
        description="TARGET_UNHEALTHY: push_up (safe) should score far above bicep_curl (risky)",
    )


def scenario_new_user() -> FakeScenario:
    """
    New user — no performance history, all body regions rated 4/5 (healthy).
    ALL_HEALTHY scenario with no history → every exercise gets personal_score = 0.50.
    """
    uid = "user_new"
    health = {BodyRegion.UPPER: 4, BodyRegion.CORE: 4, BodyRegion.LOWER: 4}
    return FakeScenario(
        user_id=uid,
        health=HealthStatus(uid, health),
        target_region=BodyRegion.UPPER,
        history=[],
        feedbacks=[],
        description="New user, no history: all personal scores should be 0.50",
    )


ALL_SCENARIOS = [
    scenario_all_healthy,
    scenario_core_pain,
    scenario_upper_injured,
    scenario_new_user,
]
