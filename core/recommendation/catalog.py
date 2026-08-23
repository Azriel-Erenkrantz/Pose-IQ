"""
Exercise metadata for the recommendation ranker.

Only the 4 exercises the app can actually track and coach
(data/exercises_seed.json) — recommending an exercise with no pose model
behind it would be a dead end for the user.
"""
from __future__ import annotations

from typing import List

from ..app_model import BodyRegion, Exercise

CATALOG: List[Exercise] = [
    Exercise("squat", "Squat",
             BodyRegion.LOWER, [BodyRegion.LOWER, BodyRegion.CORE],
             0.45, "Bodyweight or barbell squat", ["legs", "glutes"], []),
    Exercise("lunge", "Lunge",
             BodyRegion.LOWER, [BodyRegion.LOWER],
             0.40, "Forward or reverse lunge", ["legs", "glutes"], []),
    Exercise("biceps_curl", "Bicep Curl",
             BodyRegion.UPPER, [BodyRegion.UPPER],
             0.30, "Dumbbell bicep curl", ["pull", "arms"], []),
    Exercise("shoulder_press", "Shoulder Press",
             BodyRegion.UPPER, [BodyRegion.UPPER, BodyRegion.CORE],
             0.60, "Overhead shoulder press", ["push", "shoulders"], []),
]
