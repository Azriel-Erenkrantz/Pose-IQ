"""
Rating-based exercise ranker.

One tiny linear regression per exercise, trained on nothing but ratings:
    predicted(E) = bias_E + weights_E . [rating(other_1), rating(other_2), rating(other_3)]

Training data comes entirely from Mongo's 'ratings' collection — every
user's ratings, shaping both the learned weights and each user's own
predictions. Trained fresh once per server process (api/app.py, on first
request), not shipped as a fixed committed artifact.

No numpy/scikit-learn: everything here is plain Python, so it runs
identically on the lightweight deployed API (which doesn't install ML
libraries) as it does locally.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from ..app_model import Exercise, ExerciseRecommendation
from .catalog import CATALOG

EXERCISE_IDS = [e.exercise_id for e in CATALOG]

# Gitignored — a runtime cache of the Mongo-trained model, regenerated on
# every server start. See train_ranker.py to regenerate it manually.
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'recommendation_ranker.json')
)

_model_cache: Optional[dict] = None


# ── Training ───────────────────────────────────────────────────────────────────

def _mean_ratings(ratings: List[Dict[str, int]]) -> Dict[str, float]:
    """Average rating per exercise across every user who rated it."""
    sums = {ex: 0.0 for ex in EXERCISE_IDS}
    counts = {ex: 0 for ex in EXERCISE_IDS}
    for user in ratings:
        for ex, r in user.items():
            sums[ex] += r
            counts[ex] += 1
    return {ex: sums[ex] / counts[ex] if counts[ex] else 3.0 for ex in EXERCISE_IDS}


def _fit(X: List[List[float]], y: List[float], epochs: int, lr: float) -> Tuple[List[float], float]:
    """Batch gradient descent on mean squared error. Returns (weights, bias)."""
    weights = [0.0] * len(X[0])
    bias = 0.0
    n = len(X)

    for _ in range(epochs):
        errors = [bias + sum(w * x for w, x in zip(weights, row)) - actual
                  for row, actual in zip(X, y)]
        weights = [w - lr * sum(e * row[j] for e, row in zip(errors, X)) / n
                   for j, w in enumerate(weights)]
        bias -= lr * sum(errors) / n

    return weights, bias


def train(ratings: Optional[List[Dict[str, int]]] = None, epochs: int = 3000, lr: float = 0.01) -> dict:
    """Fit one regression per exercise — on Mongo's ratings by default —
    and persist the result to MODEL_PATH as plain JSON."""
    if ratings is None:
        from . import ratings_service
        ratings = ratings_service.get_all_ratings_for_training()
    means = _mean_ratings(ratings)

    models: Dict[str, dict] = {}
    for target in EXERCISE_IDS:
        others = [ex for ex in EXERCISE_IDS if ex != target]
        rows = [(user, user[target]) for user in ratings if target in user]

        if rows:
            X = [[user.get(o, means[o]) for o in others] for user, _ in rows]
            y = [float(rating) for _, rating in rows]
            weights, bias = _fit(X, y, epochs, lr)
        else:
            # Nobody has rated this exercise (e.g. Mongo not seeded yet) —
            # predict the mean rather than crashing on an empty training set.
            weights, bias = [0.0] * len(others), means[target]

        models[target] = {"others": others, "weights": weights, "bias": bias}

    model = {"means": means, "models": models}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=2)

    # Update the live cache too — without this, a second train() call in a
    # process that already answered a request would keep serving the old
    # cached model until the process restarts.
    global _model_cache
    _model_cache = model
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def _load_model() -> dict:
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, encoding='utf-8') as f:
            _model_cache = json.load(f)
    return _model_cache


def _predict(exercise_id: str, user_ratings: Dict[str, int], model: dict) -> float:
    spec = model["models"][exercise_id]
    features = [user_ratings.get(o, model["means"][o]) for o in spec["others"]]
    raw = spec["bias"] + sum(w * x for w, x in zip(spec["weights"], features))
    return max(1.0, min(5.0, raw))


def recommend_for_user(user_ratings: Dict[str, int], exercises: List[Exercise]) -> List[ExerciseRecommendation]:
    """Rank the given exercises for one user, using only their own ratings
    (on the *other* exercises) and the trained model."""
    model = _load_model()
    results = []

    for exercise in exercises:
        ex_id = exercise.exercise_id
        if ex_id in user_ratings:
            # Already rated — show that rating, not a prediction from
            # other exercises they may not have rated.
            rating = user_ratings[ex_id]
            reason, reason_code, reason_params = f"You rated this {rating}/5", 'rated_by_you', {'rating': float(rating)}
        else:
            rating = _predict(ex_id, user_ratings, model)
            if user_ratings:
                reason, reason_code, reason_params = "Predicted from your other ratings", 'predicted_from_ratings', {}
            else:
                reason, reason_code, reason_params = "New to you — based on other users' ratings", 'no_ratings_yet', {}

        results.append(ExerciseRecommendation(
            exercise=exercise,
            score=round((rating - 1.0) / 4.0, 3),  # 1-5 -> 0-1
            reason=reason,
            reason_code=reason_code,
            reason_params=reason_params,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
