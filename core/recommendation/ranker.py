"""
Rating-based exercise ranker.

One tiny linear regression per exercise, trained on two signals per *other*
exercise: the user's own 1-5 rating, and their average session form score —
how much they say they like it, and how well they actually do it:
    predicted(E) = bias_E + weights_E . [rating(other_1..3), avg_score(other_1..3)]

Training data comes entirely from Mongo — the 'ratings' collection and the
'sessions' collection's per-session overall_score — every user's data
shaping both the learned weights and each user's own predictions. Trained
fresh once per server process (api/app.py, on first request), not shipped
as a fixed committed artifact.

No numpy/scikit-learn: everything here is plain Python, so it runs
identically on the lightweight deployed API (which doesn't install ML
libraries) as it does locally.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from ..app_model import Exercise, ExerciseRecommendation
from .catalog import CATALOG

EXERCISE_IDS = [e.exercise_id for e in CATALOG]

# Neutral fallback for an exercise a user (or the whole population) has no
# session data for yet — the midpoint of the 0-100 form-score scale.
DEFAULT_SCORE = 75.0

# Gitignored — a runtime cache of the Mongo-trained model, regenerated on
# every server start. See train_ranker.py to regenerate it manually.
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'recommendation_ranker.json')
)

_model_cache: Optional[dict] = None


# ── Training ───────────────────────────────────────────────────────────────────

def _mean_values(per_user: Iterable[Dict[str, float]], default: float) -> Dict[str, float]:
    """Average value per exercise across every user who has one — used for
    both ratings (1-5) and average form scores (0-100), just with a
    different fallback default for an exercise nobody has data for yet."""
    sums = {ex: 0.0 for ex in EXERCISE_IDS}
    counts = {ex: 0 for ex in EXERCISE_IDS}
    for values in per_user:
        for ex, v in values.items():
            sums[ex] += v
            counts[ex] += 1
    return {ex: sums[ex] / counts[ex] if counts[ex] else default for ex in EXERCISE_IDS}


# Scores are 0-100 vs. ratings' 1-5 — feeding both into gradient descent
# unscaled makes the score features' gradients ~20x larger than the rating
# features', which destabilizes training at the learning rate tuned for
# ratings alone. Rescaling to the same ballpark keeps a single shared
# learning rate valid for every feature.
SCORE_TO_RATING_SCALE = 20.0


def _features(
    user_ratings: Dict[str, int], user_scores: Dict[str, float], others: List[str],
    means: Dict[str, float], score_means: Dict[str, float],
) -> List[float]:
    """One user's feature vector for predicting a target exercise: their
    rating on each *other* exercise, followed by their average performance
    score on those same exercises (rescaled onto the same 1-5 ballpark as
    ratings) — an exercise they haven't rated or performed falls back to
    the population mean/mean score."""
    return (
        [user_ratings.get(o, means[o]) for o in others] +
        [user_scores.get(o, score_means[o]) / SCORE_TO_RATING_SCALE for o in others]
    )


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


def train(
    ratings: Optional[Dict[str, Dict[str, int]]] = None,
    avg_scores: Optional[Dict[str, Dict[str, float]]] = None,
    epochs: int = 3000, lr: float = 0.01,
) -> dict:
    """Fit one regression per exercise — on Mongo's ratings and average
    session scores by default — and persist the result to MODEL_PATH as
    plain JSON."""
    if ratings is None:
        from . import ratings_service
        ratings = ratings_service.get_all_ratings_for_training()
    if avg_scores is None:
        from . import scores_service
        avg_scores = scores_service.get_all_avg_scores_for_training()

    means = _mean_values(ratings.values(), default=3.0)
    score_means = _mean_values(avg_scores.values(), default=DEFAULT_SCORE)

    models: Dict[str, dict] = {}
    for target in EXERCISE_IDS:
        others = [ex for ex in EXERCISE_IDS if ex != target]
        rows = [
            (user_ratings, avg_scores.get(user_id, {}), user_ratings[target])
            for user_id, user_ratings in ratings.items()
            if target in user_ratings
        ]

        if rows:
            X = [_features(r, s, others, means, score_means) for r, s, _ in rows]
            y = [float(rating) for _, _, rating in rows]
            weights, bias = _fit(X, y, epochs, lr)
        else:
            # Nobody has rated this exercise (e.g. Mongo not seeded yet) —
            # predict the mean rather than crashing on an empty training set.
            weights, bias = [0.0] * (2 * len(others)), means[target]

        models[target] = {"others": others, "weights": weights, "bias": bias}

    model = {"means": means, "score_means": score_means, "models": models}
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


def _predict(exercise_id: str, user_ratings: Dict[str, int], user_scores: Dict[str, float], model: dict) -> float:
    spec = model["models"][exercise_id]
    features = _features(user_ratings, user_scores, spec["others"], model["means"], model["score_means"])
    raw = spec["bias"] + sum(w * x for w, x in zip(spec["weights"], features))
    return max(1.0, min(5.0, raw))


def recommend_for_user(
    user_ratings: Dict[str, int],
    exercises: List[Exercise],
    user_scores: Optional[Dict[str, float]] = None,
) -> List[ExerciseRecommendation]:
    """Rank the given exercises for one user, using only their own ratings
    and average performance scores (on the *other* exercises) and the
    trained model."""
    model = _load_model()
    user_scores = user_scores or {}
    results = []

    for exercise in exercises:
        ex_id = exercise.exercise_id
        if ex_id in user_ratings:
            # Already rated — show that rating, not a prediction from
            # other exercises they may not have rated.
            rating = user_ratings[ex_id]
            reason, reason_code, reason_params = f"You rated this {rating}/5", 'rated_by_you', {'rating': float(rating)}
        else:
            rating = _predict(ex_id, user_ratings, user_scores, model)
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
