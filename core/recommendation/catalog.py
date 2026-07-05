from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import uuid

from ..app_model import BodyRegion, Exercise, CommunityRecord

# ---------------------------------------------------------------------------
# Exercise catalog
# IDs for squat / plank / lunge / bicep_curl / shoulder_press match the
# existing data/exercises.json used by the pose-detection system.
# ---------------------------------------------------------------------------

CATALOG: List[Exercise] = [
    # ── UPPER ────────────────────────────────────────────────────────────
    Exercise(
        "bicep_curl", "Bicep Curl",
        BodyRegion.UPPER, [BodyRegion.UPPER],
        0.30, "Dumbbell bicep curl", ["pull", "arms"],
    ),
    Exercise(
        "shoulder_press", "Shoulder Press",
        BodyRegion.UPPER, [BodyRegion.UPPER, BodyRegion.CORE],
        0.60, "Overhead shoulder press", ["push", "shoulders"],
    ),
    Exercise(
        "push_up", "Push-up",
        BodyRegion.UPPER, [BodyRegion.UPPER, BodyRegion.CORE],
        0.40, "Classic push-up", ["push", "chest"],
    ),
    Exercise(
        "tricep_dip", "Tricep Dip",
        BodyRegion.UPPER, [BodyRegion.UPPER],
        0.50, "Parallel bar tricep dip", ["push", "arms"],
    ),
    # ── CORE ─────────────────────────────────────────────────────────────
    Exercise(
        "plank", "Plank",
        BodyRegion.CORE, [BodyRegion.CORE, BodyRegion.UPPER],
        0.35, "Static plank hold", ["isometric", "core"],
    ),
    Exercise(
        "crunch", "Crunch",
        BodyRegion.CORE, [BodyRegion.CORE],
        0.25, "Basic abdominal crunch", ["core", "abs"],
    ),
    Exercise(
        "russian_twist", "Russian Twist",
        BodyRegion.CORE, [BodyRegion.CORE],
        0.50, "Seated oblique rotation", ["core", "obliques"],
    ),
    Exercise(
        "leg_raise", "Leg Raise",
        BodyRegion.CORE, [BodyRegion.CORE, BodyRegion.LOWER],
        0.55, "Lying or hanging leg raise", ["core", "hip flexors"],
    ),
    # ── LOWER ────────────────────────────────────────────────────────────
    Exercise(
        "squat", "Squat",
        BodyRegion.LOWER, [BodyRegion.LOWER, BodyRegion.CORE],
        0.45, "Bodyweight or barbell squat", ["legs", "glutes"],
    ),
    Exercise(
        "lunge", "Lunge",
        BodyRegion.LOWER, [BodyRegion.LOWER],
        0.40, "Forward or reverse lunge", ["legs", "glutes"],
    ),
    Exercise(
        "deadlift", "Deadlift",
        BodyRegion.LOWER, [BodyRegion.LOWER, BodyRegion.CORE, BodyRegion.UPPER],
        0.80, "Conventional deadlift", ["legs", "back", "full body"],
    ),
    Exercise(
        "calf_raise", "Calf Raise",
        BodyRegion.LOWER, [BodyRegion.LOWER],
        0.20, "Standing calf raise", ["legs", "calves"],
    ),
    Exercise(
        "glute_bridge", "Glute Bridge",
        BodyRegion.LOWER, [BodyRegion.LOWER, BodyRegion.CORE],
        0.30, "Lying hip extension", ["glutes", "core"],
    ),
]


def get_by_id(exercise_id: str) -> Optional[Exercise]:
    return next((e for e in CATALOG if e.exercise_id == exercise_id), None)


def get_by_region(region: BodyRegion) -> List[Exercise]:
    return [e for e in CATALOG if e.primary_region == region]


# ---------------------------------------------------------------------------
# Mock community data
# Represents aggregated feedback from past users with various health profiles.
# In production this comes from the database.
# ---------------------------------------------------------------------------

COMMUNITY_DATA: List[CommunityRecord] = [
    # Users with core pain who found push-ups comfortable
    CommunityRecord(str(uuid.uuid4()), "push_up",
                    {BodyRegion.UPPER: 5, BodyRegion.CORE: 2, BodyRegion.LOWER: 4}, 0.72,
                    datetime(2026, 5, 10)),
    CommunityRecord(str(uuid.uuid4()), "push_up",
                    {BodyRegion.UPPER: 4, BodyRegion.CORE: 2, BodyRegion.LOWER: 5}, 0.75,
                    datetime(2026, 5, 18)),
    CommunityRecord(str(uuid.uuid4()), "push_up",
                    {BodyRegion.UPPER: 5, BodyRegion.CORE: 3, BodyRegion.LOWER: 4}, 0.65,
                    datetime(2026, 6, 1)),
    # Users with core pain who found shoulder press uncomfortable
    CommunityRecord(str(uuid.uuid4()), "shoulder_press",
                    {BodyRegion.UPPER: 5, BodyRegion.CORE: 2, BodyRegion.LOWER: 4}, 0.30,
                    datetime(2026, 5, 12)),
    CommunityRecord(str(uuid.uuid4()), "shoulder_press",
                    {BodyRegion.UPPER: 4, BodyRegion.CORE: 1, BodyRegion.LOWER: 5}, 0.20,
                    datetime(2026, 5, 20)),
    # Fully healthy users who love challenging squats
    CommunityRecord(str(uuid.uuid4()), "squat",
                    {BodyRegion.UPPER: 5, BodyRegion.CORE: 5, BodyRegion.LOWER: 5}, 0.90,
                    datetime(2026, 5, 15)),
    CommunityRecord(str(uuid.uuid4()), "squat",
                    {BodyRegion.UPPER: 4, BodyRegion.CORE: 4, BodyRegion.LOWER: 4}, 0.85,
                    datetime(2026, 5, 25)),
    # Users with lower body pain who found calf raises safe
    CommunityRecord(str(uuid.uuid4()), "calf_raise",
                    {BodyRegion.UPPER: 4, BodyRegion.CORE: 4, BodyRegion.LOWER: 2}, 0.88,
                    datetime(2026, 6, 3)),
    CommunityRecord(str(uuid.uuid4()), "calf_raise",
                    {BodyRegion.UPPER: 5, BodyRegion.CORE: 3, BodyRegion.LOWER: 1}, 0.80,
                    datetime(2026, 6, 8)),
    # Users with upper body injury who rated bicep_curl low (aggravates it)
    CommunityRecord(str(uuid.uuid4()), "bicep_curl",
                    {BodyRegion.UPPER: 2, BodyRegion.CORE: 4, BodyRegion.LOWER: 5}, 0.15,
                    datetime(2026, 6, 5)),
]
