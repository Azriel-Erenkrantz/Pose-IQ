"""
Synthetic user ratings the ranker trains on — not real data.

Two fake archetypes give the ratings real structure to learn from:
  - "lower-body" fans rate squat/lunge high, biceps_curl/shoulder_press low
  - "upper-body" fans rate biceps_curl/shoulder_press high, squat/lunge low
Each fake user is one archetype plus a bit of per-user noise, so ratings
aren't identical within a group but the group pattern still comes through.

FAKE_RATINGS: one row per fake user, exercise_id -> rating (1-5). Not every
user rates every exercise, matching how a real user would leave gaps.
"""
from __future__ import annotations

import random
from typing import Dict, List

EXERCISE_IDS = ["squat", "lunge", "biceps_curl", "shoulder_press"]


def _lower_body_fan(rng: random.Random) -> Dict[str, int]:
    ratings = {
        "squat":          rng.randint(4, 5),
        "lunge":           rng.randint(4, 5),
        "biceps_curl":     rng.randint(1, 3),
        "shoulder_press":  rng.randint(1, 3),
    }
    if rng.random() < 0.3:
        del ratings[rng.choice(list(ratings))]
    return ratings


def _upper_body_fan(rng: random.Random) -> Dict[str, int]:
    ratings = {
        "squat":          rng.randint(1, 3),
        "lunge":           rng.randint(1, 3),
        "biceps_curl":     rng.randint(4, 5),
        "shoulder_press":  rng.randint(4, 5),
    }
    if rng.random() < 0.3:
        del ratings[rng.choice(list(ratings))]
    return ratings


def _generate(seed: int = 42, n_per_archetype: int = 10) -> List[Dict[str, int]]:
    rng = random.Random(seed)
    users = [_lower_body_fan(rng) for _ in range(n_per_archetype)]
    users += [_upper_body_fan(rng) for _ in range(n_per_archetype)]
    rng.shuffle(users)
    return users


FAKE_RATINGS: List[Dict[str, int]] = _generate()
