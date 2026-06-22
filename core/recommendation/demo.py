"""
Runnable demo for the exercise recommendation system.

Each scenario is triggered by calling a simulation function that represents a
button press or form submission in the (not yet built) UI.

Run with:
    python -m core.recommendation.demo
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from core.recommendation.models import BodyRegion, ExercisePerformanceRecord
from core.recommendation.simulation import (
    change_target_region,
    complete_exercise,
    print_recommendations,
    seed_history,
    start_session,
    submit_feedback,
    update_health_status,
)


# ── Pre-built history ───────────────────────────────────────────────────────
# Represents what the performance-monitoring model has collected over past weeks.

def _make_record(user_id, exercise_id, difficulty, form, violations, days_ago=7):
    return ExercisePerformanceRecord(
        id=str(uuid.uuid4()),
        exercise_id=exercise_id,
        user_id=user_id,
        session_id="seed",
        timestamp=datetime.now() - timedelta(days=days_ago),
        difficulty_score=difficulty,
        form_quality_score=form,
        completion_rate=round(max(0.1, 1.0 - difficulty * 0.3), 2),
        violations=violations,
        reps_completed=max(1, int((1.0 - difficulty) * 12) + 3),
        sets_completed=3,
        health_snapshot={BodyRegion.UPPER: 5, BodyRegion.CORE: 5, BodyRegion.LOWER: 5},
    )


def _seed_alex(user_id: str) -> None:
    """Alex: shoulder press was hard, bicep curl was easy, push-up was moderate."""
    seed_history(user_id, [
        _make_record(user_id, "shoulder_press", 0.82, 0.55, ["incomplete_range_of_motion"], 14),
        _make_record(user_id, "shoulder_press", 0.79, 0.60, ["speed_too_fast"], 7),
        _make_record(user_id, "shoulder_press", 0.84, 0.58, ["incomplete_range_of_motion"], 3),
        _make_record(user_id, "bicep_curl",     0.28, 0.92, [], 12),
        _make_record(user_id, "bicep_curl",     0.32, 0.90, [], 5),
        _make_record(user_id, "push_up",        0.46, 0.80, ["minor_form_break"], 10),
        _make_record(user_id, "push_up",        0.40, 0.85, [], 4),
        _make_record(user_id, "squat",          0.72, 0.68, ["knee_cave"], 9),
        _make_record(user_id, "squat",          0.68, 0.72, ["knee_cave"], 2),
    ])


# ── Demo ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 62)
    print("      POSE-IQ — EXERCISE RECOMMENDATION SYSTEM DEMO")
    print("=" * 62)

    user = "user_alex"
    _seed_alex(user)

    # ── SCENARIO 1: All healthy — train upper body ──────────────────
    print("\n[SCENARIO 1]  Alex is fully healthy. Training: UPPER BODY.")
    print("  Expectation: shoulder_press (historically hard) ranks highest —")
    print("  now at full health is the time to tackle weaknesses.")

    session = start_session(
        user_id=user,
        target_region=BodyRegion.UPPER,
        health_ratings={
            BodyRegion.UPPER: 5,
            BodyRegion.CORE:  5,
            BodyRegion.LOWER: 5,
        },
    )
    print_recommendations(session)

    # ── SCENARIO 2: Core pain — adjacent region affected ───────────
    print("[SCENARIO 2]  [Button press] Alex reports core pain (2/5).")
    print("  Expectation: shoulder_press (hard + uses core) drops sharply.")
    print("  bicep_curl and push_up (moderate, manageable) rise.")

    session = update_health_status(session.id, BodyRegion.CORE, 2)
    print_recommendations(session)

    # ── Complete an exercise and give feedback ──────────────────────
    print("[Button press] Alex selects Push-up and finishes the set.")
    record = complete_exercise(
        session.id, "push_up",
        difficulty=0.38,
        form_quality=0.87,
        violations=[],
    )
    print(f"  Performance monitor returned:")
    print(f"    difficulty  : {record.difficulty_score:.0%}")
    print(f"    form quality: {record.form_quality_score:.0%}")
    print(f"    reps / sets : {record.reps_completed} x {record.sets_completed}")
    print(f"    violations  : {record.violations or 'none'}")

    feedback = submit_feedback(session.id, "push_up", rating=4, comment="Felt comfortable")
    print(f"\n[Button press] Alex submits feedback: {feedback.rating}/5 stars")
    print("  Recommendations after feedback (push_up score nudged up):")
    print_recommendations(session)

    # ── SCENARIO 3: Upper body injured ─────────────────────────────
    print("[SCENARIO 3]  [Button press] Alex reports upper body pain (2/5).")
    print("  Expectation: only exercises with good form history and low")
    print("  difficulty survive. shoulder_press (poor form + hard) drops to ~0.")

    session = update_health_status(session.id, BodyRegion.UPPER, 2)
    print_recommendations(session)

    # ── SCENARIO 4: Switch to lower body ───────────────────────────
    print("[SCENARIO 4]  [Button press] Alex switches to LOWER BODY training.")
    print("  Core is still 2/5 (adjacent to lower). Expectation: squat")
    print("  (historically challenging) is penalised; easier exercises rise.")

    session = change_target_region(session.id, BodyRegion.LOWER)
    print_recommendations(session)

    print("=" * 62)
    print("Demo complete.")
    print()


if __name__ == "__main__":
    run()
