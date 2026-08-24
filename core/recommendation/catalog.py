"""
Exercise metadata for the recommendation ranker.

Only the 4 exercises the app can actually track and coach
(data/exercises_seed.json) — recommending an exercise with no pose model
behind it would be a dead end for the user.
"""
from __future__ import annotations

from typing import List

from ..app_model import Exercise

CATALOG: List[Exercise] = [
    Exercise("squat", "Squat", "Bodyweight or barbell squat", []),
    Exercise("lunge", "Lunge", "Forward or reverse lunge", []),
    Exercise("biceps_curl", "Bicep Curl", "Dumbbell bicep curl", []),
    Exercise("shoulder_press", "Shoulder Press", "Overhead shoulder press", []),
]
