"""
Linear regression ranker for exercise recommendations.

Replaces rule-based personal_score inside recommender.py when:
  1. Enough total sessions exist  (>= MIN_SESSIONS_TOTAL)
  2. A trained model file exists  (MODEL_PATH)

Workflow
--------
Training  (offline, call once enough data exists):
    from core.recommendation.ml_ranker import train
    train(all_user_history, all_user_feedbacks)

Inference (automatic, called inside recommender._personal_score):
    score = predict(exercise_records, current_health)

Falls back to rule-based transparently when either condition is unmet.

Feature vector  (11 dimensions per (user, exercise) pair)
--------------
avg_difficulty, avg_form, avg_completion, avg_violations,
form_trend, difficulty_trend,
sessions_count_norm, recency_norm,
upper_health_norm, core_health_norm, lower_health_norm

Satisfaction label  (composite 0–1, used as training target y)
------------------
0.30 * form_improvement_signal
0.25 * avg_completion_rate
0.25 * return_signal          (how many times they came back)
0.20 * explicit_feedback      (star rating if available, else 0.5)
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import List, Optional

from .models import BodyRegion, ExercisePerformanceRecord, HealthStatus, UserFeedback

# ── Configuration (monkeypatchable in tests) ──────────────────────────────────

MIN_SESSIONS_TOTAL       = 20   # total sessions before ML activates
MIN_SESSIONS_PER_EXERCISE = 2   # minimum per exercise to include in training set

MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'recommendation_ranker.pkl')
)


# ── Feature engineering ───────────────────────────────────────────────────────

def _feature_vector(
    records: List[ExercisePerformanceRecord],
    health: HealthStatus,
) -> Optional[List[float]]:
    """
    11-dimensional feature vector for one (user, exercise) pair.
    Returns None when no history exists — caller falls back to rule-based.
    """
    if not records:
        return None

    by_time = sorted(records, key=lambda r: r.timestamp)
    recent  = by_time[-5:]

    avg_diff       = sum(r.difficulty_score   for r in recent) / len(recent)
    avg_form       = sum(r.form_quality_score for r in recent) / len(recent)
    avg_completion = sum(r.completion_rate    for r in recent) / len(recent)
    avg_violations = sum(len(r.violations)    for r in recent) / len(recent)

    if len(by_time) >= 2:
        n          = len(by_time)
        form_trend = (by_time[-1].form_quality_score - by_time[0].form_quality_score) / n
        diff_trend = (by_time[-1].difficulty_score   - by_time[0].difficulty_score)   / n
    else:
        form_trend = diff_trend = 0.0

    sessions_norm = min(len(records) / 10.0, 1.0)
    days_since    = (datetime.now() - recent[-1].timestamp).days
    recency_norm  = max(0.0, 1.0 - days_since / 30.0)   # 1 = today, 0 = 30+ days ago

    return [
        avg_diff,
        avg_form,
        avg_completion,
        min(avg_violations / 3.0, 1.0),
        form_trend,
        diff_trend,
        sessions_norm,
        recency_norm,
        health.get(BodyRegion.UPPER) / 5.0,
        health.get(BodyRegion.CORE)  / 5.0,
        health.get(BodyRegion.LOWER) / 5.0,
    ]


# ── Satisfaction label ────────────────────────────────────────────────────────

def _satisfaction_score(
    records: List[ExercisePerformanceRecord],
    feedbacks: List[UserFeedback],
) -> Optional[float]:
    """
    Composite satisfaction label (0–1) used as the training target y.

    Captures what the user actually told us through behaviour and ratings:
      - Are they improving?        (form trend)
      - Did they finish?           (completion)
      - Did they come back?        (return signal)
      - Did they say they liked it? (explicit rating)
    """
    if not records:
        return None

    by_time = sorted(records, key=lambda r: r.timestamp)

    # Form improvement: positive trend → signal approaches 1.0
    if len(by_time) >= 2:
        form_vals   = [r.form_quality_score for r in by_time]
        trend       = (form_vals[-1] - form_vals[0]) / len(form_vals)
        form_signal = min(1.0, max(0.0, trend * 5 + 0.5))  # centre at 0, scale up
    else:
        form_signal = by_time[0].form_quality_score

    completion_signal = sum(r.completion_rate for r in records) / len(records)
    return_signal     = min(len(records) / 5.0, 1.0)   # capped: 5+ sessions = 1.0

    ex_id    = records[0].exercise_id
    ex_fb    = [f for f in feedbacks if f.exercise_id == ex_id]
    feedback_signal = (
        sum(f.rating / 5.0 for f in ex_fb) / len(ex_fb)
        if ex_fb else 0.5
    )

    return round(
        0.30 * form_signal +
        0.25 * completion_signal +
        0.25 * return_signal +
        0.20 * feedback_signal,
        3,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def has_enough_data(history: List[ExercisePerformanceRecord]) -> bool:
    """True when total session count meets the activation threshold."""
    return len(history) >= MIN_SESSIONS_TOTAL


def model_exists(model_path: Optional[str] = None) -> bool:
    return os.path.exists(model_path or MODEL_PATH)


def train(
    history: List[ExercisePerformanceRecord],
    feedbacks: List[UserFeedback],
    model_path: Optional[str] = None,
) -> bool:
    """
    Build and save the ranker from accumulated session history.

    Returns True on success, False if not enough labelled examples could be
    extracted (need at least 3 exercises with MIN_SESSIONS_PER_EXERCISE each).
    Raises ImportError if scikit-learn is not installed.
    """
    from sklearn.linear_model import Ridge

    exercise_ids = {r.exercise_id for r in history}
    X: List[List[float]] = []
    y: List[float]       = []

    for ex_id in exercise_ids:
        ex_records = [r for r in history if r.exercise_id == ex_id]
        if len(ex_records) < MIN_SESSIONS_PER_EXERCISE:
            continue

        health = HealthStatus("_train", ex_records[-1].health_snapshot)
        features = _feature_vector(ex_records, health)
        label    = _satisfaction_score(ex_records, feedbacks)

        if features is not None and label is not None:
            X.append(features)
            y.append(label)

    if len(X) < 3:
        return False

    model = Ridge()
    model.fit(X, y)

    path = model_path or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

    return True


def predict(
    records: List[ExercisePerformanceRecord],
    health: HealthStatus,
    model_path: Optional[str] = None,
) -> Optional[float]:
    """
    Predict satisfaction score for one exercise given the user's history on it.
    Returns None when there is no history (caller falls back to rule-based 0.50).
    Score is clamped to [0, 1].
    """
    features = _feature_vector(records, health)
    if features is None:
        return None

    path = model_path or MODEL_PATH
    with open(path, 'rb') as f:
        model = pickle.load(f)

    score = float(model.predict([features])[0])
    return round(min(1.0, max(0.0, score)), 3)
