"""
Storage for the user's own exercise ratings.

Collection 'ratings' — one document per (user_id, exercise_id), upserted
on every re-rate. This is the only real (non-fake) data the ranker ever
sees, and only at inference time — never used to retrain the model.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

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
