"""
Read-only access to users' average per-exercise form scores.

Derived from the 'sessions' collection (each session's overall_score) — used
by the ranker as a second feature alongside ratings, since how well someone
actually performs an exercise is a different signal than how much they say
they like it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

from core.db import get_db


def get_user_avg_scores(user_id: str) -> Dict[str, float]:
    """exercise_id -> average overall_score, across this user's sessions."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for doc in get_db().sessions.find({'user_id': user_id}, {'exercise_id': 1, 'overall_score': 1}):
        sums[doc['exercise_id']] += doc['overall_score']
        counts[doc['exercise_id']] += 1
    return {ex: sums[ex] / counts[ex] for ex in sums}


def get_all_avg_scores_for_training() -> Dict[str, Dict[str, float]]:
    """Every user's average score per exercise, keyed by user_id — the
    shape ranker.train() needs to join against each user's ratings."""
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for doc in get_db().sessions.find({}, {'user_id': 1, 'exercise_id': 1, 'overall_score': 1}):
        user_id, exercise_id = doc['user_id'], doc['exercise_id']
        sums[user_id][exercise_id] += doc['overall_score']
        counts[user_id][exercise_id] += 1
    return {
        user_id: {ex: sums[user_id][ex] / counts[user_id][ex] for ex in exercises}
        for user_id, exercises in sums.items()
    }
