"""
Rating-based exercise ranker.

One tiny linear regression per exercise, trained on nothing but ratings:
    predicted(E) = bias_E + weights_E . [rating(other_1), rating(other_2), rating(other_3)]

Training data comes entirely from Mongo's 'ratings' collection
(ratings_service.get_all_ratings_for_training) — every user's ratings,
shaping both the learned weights and each user's own predictions. Trained
fresh once per server process (api/app.py, on first request) rather than
shipped as a fixed committed artifact.

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

# Lives under the gitignored data/models/ — this is now a runtime-generated
# cache of the Mongo-trained model, regenerated on every server start, not a
# fixed artifact to commit. See train_ranker.py to regenerate it manually.
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'recommendation_ranker.json')
)


# ── Training ───────────────────────────────────────────────────────────────────

def _mean_ratings(ratings: List[Dict[str, int]]) -> Dict[str, float]:
    """Average rating per exercise across every user who rated it."""
    sums = {ex: 0.0 for ex in EXERCISE_IDS}
    counts = {ex: 0 for ex in EXERCISE_IDS}
    for user in ratings:
        for ex, r in user.items():
            sums[ex] += r
            counts[ex] += 1
    return {ex: (sums[ex] / counts[ex] if counts[ex] else 3.0) for ex in EXERCISE_IDS}


def _train_one(X: List[List[float]], y: List[float], epochs: int, lr: float) -> Tuple[List[float], float]:
    """Batch gradient descent on mean squared error. Returns (weights, bias)."""
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    n = len(X)

    for _ in range(epochs):
        predictions = [bias + sum(w * x for w, x in zip(weights, row)) for row in X]
        errors = [pred - actual for pred, actual in zip(predictions, y)]

        # d(MSE)/dw_j = mean(error * x_j); d(MSE)/dbias = mean(error)
        grad_weights = [sum(e * row[j] for e, row in zip(errors, X)) / n for j in range(n_features)]
        grad_bias = sum(errors) / n

        weights = [w - lr * g for w, g in zip(weights, grad_weights)]
        bias -= lr * grad_bias

    return weights, bias


def train(ratings: Optional[List[Dict[str, int]]] = None, epochs: int = 3000, lr: float = 0.01) -> dict:
    """
    Fit one regression per exercise on the given ratings — pulled from
    Mongo by default — and persist the result to MODEL_PATH as plain JSON.
    """
    if ratings is None:
        from . import ratings_service
        ratings = ratings_service.get_all_ratings_for_training()
    means = _mean_ratings(ratings)

    models: Dict[str, dict] = {}
    for target in EXERCISE_IDS:
        others = [ex for ex in EXERCISE_IDS if ex != target]

        X: List[List[float]] = []
        y: List[float] = []
        for user in ratings:
            if target not in user:
                continue  # nothing to predict for this user
            X.append([user.get(o, means[o]) for o in others])
            y.append(float(user[target]))

        if X:
            weights, bias = _train_one(X, y, epochs, lr)
        else:
            # Nobody has rated this exercise at all (e.g. Mongo not seeded
            # yet) — fall back to "always predict the mean" rather than
            # crashing on an empty training set.
            weights, bias = [0.0] * len(others), means[target]
        models[target] = {"others": others, "weights": weights, "bias": bias}

    model = {"means": means, "models": models}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=2)

    # Make the freshly trained model live immediately — without this,
    # calling train() again in a process that already served a request
    # (which lazily populates _model_cache) would silently keep serving
    # the stale cached model until the process restarts.
    global _model_cache
    _model_cache = model
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

_model_cache: Optional[dict] = None


def _load_model() -> dict:
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, encoding='utf-8') as f:
            _model_cache = json.load(f)
    return _model_cache


def _predict(exercise_id: str, user_ratings: Dict[str, int], model: dict) -> float:
    spec = model["models"][exercise_id]
    means = model["means"]
    features = [user_ratings.get(o, means[o]) for o in spec["others"]]
    raw = spec["bias"] + sum(w * x for w, x in zip(spec["weights"], features))
    return max(1.0, min(5.0, raw))


def recommend_for_user(
    user_ratings: Dict[str, int],
    exercises: List[Exercise],
) -> List[ExerciseRecommendation]:
    """
    Rank the given exercises for one user, using only that user's own
    ratings (on the *other* exercises) and the trained model.
    """
    model = _load_model()

    results: List[ExerciseRecommendation] = []
    for exercise in exercises:
        if exercise.exercise_id in user_ratings:
            # Already rated by this user — show their own rating, not a
            # prediction from other exercises they may not have rated.
            own_rating = user_ratings[exercise.exercise_id]
            score = round((own_rating - 1.0) / 4.0, 3)
            reason = f"You rated this {own_rating}/5"
            reason_code, reason_params = 'rated_by_you', {'rating': float(own_rating)}
        elif user_ratings:
            predicted = _predict(exercise.exercise_id, user_ratings, model)
            score = round((predicted - 1.0) / 4.0, 3)  # 1-5 -> 0-1
            reason = "Predicted from your other ratings"
            reason_code, reason_params = 'predicted_from_ratings', {}
        else:
            predicted = _predict(exercise.exercise_id, user_ratings, model)
            score = round((predicted - 1.0) / 4.0, 3)
            reason = "New to you — based on other users' ratings"
            reason_code, reason_params = 'no_ratings_yet', {}

        results.append(ExerciseRecommendation(
            exercise=exercise,
            score=score,
            reason=reason,
            reason_code=reason_code,
            reason_params=reason_params,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
