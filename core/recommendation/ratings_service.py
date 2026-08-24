"""
Storage for users' exercise ratings.

Collection 'ratings' — one document per (user_id, exercise_id), upserted
on every re-rate. This is the only data the recommendation ranker trains
on and predicts from.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from core.app_model import Rating
from core.db import get_db


def save_rating(user_id: str, exercise_id: str, rating: int) -> Rating:
    rated_at = datetime.now()
    get_db().ratings.update_one(
        {'user_id': user_id, 'exercise_id': exercise_id},
        {'$set': {'rating': rating, 'rated_at': rated_at}},
        upsert=True,
    )
    return Rating(user_id=user_id, exercise_id=exercise_id, rating=rating, rated_at=rated_at)


def get_user_ratings(user_id: str) -> Dict[str, int]:
    """exercise_id -> rating, for every exercise this user has rated."""
    docs = get_db().ratings.find({'user_id': user_id}, {'exercise_id': 1, 'rating': 1})
    return {doc['exercise_id']: doc['rating'] for doc in docs}


def get_all_ratings_for_training() -> Dict[str, Dict[str, int]]:
    """Every user's ratings, keyed by user_id — the shape ranker.train()
    needs to join against each user's average scores."""
    db = get_db()
    by_user: Dict[str, Dict[str, int]] = {}
    for doc in db.ratings.find():
        by_user.setdefault(doc['user_id'], {})[doc['exercise_id']] = doc['rating']
    return by_user
