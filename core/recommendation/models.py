from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import uuid


class BodyRegion(str, Enum):
    UPPER = "upper"
    CORE = "core"
    LOWER = "lower"


class HealthScenario(str, Enum):
    ALL_HEALTHY = "all_healthy"
    TARGET_HEALTHY_ADJACENT_NOT = "target_healthy_adjacent_not"
    TARGET_UNHEALTHY = "target_unhealthy"


@dataclass
class HealthStatus:
    user_id: str
    ratings: Dict[BodyRegion, int]  # 1 = poor, 5 = fully healthy
    recorded_at: datetime = field(default_factory=datetime.now)

    def get(self, region: BodyRegion) -> int:
        return self.ratings.get(region, 3)


@dataclass
class CatalogExercise:
    """
    Recommendation-layer representation of an exercise.
    Distinct from the pose-detection Exercise in core/exercise/exercise_model.py.
    exercise_id matches the ID used by the pose detection system when the exercise exists there.
    """
    exercise_id: str
    name: str
    primary_region: BodyRegion
    body_regions: List[BodyRegion]
    base_difficulty: float   # 0–1, inherent difficulty of the movement
    description: str
    tags: List[str] = field(default_factory=list)


@dataclass
class ExercisePerformanceRecord:
    """
    Output contract for the performance-monitoring ML model.
    The model is under construction; this dataclass defines what it must return.
    """
    id: str
    exercise_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    difficulty_score: float    # 0–1: how hard this was for the user
    form_quality_score: float  # 0–1: how correct their form was
    completion_rate: float     # 0–1: fraction of target reps/duration completed
    violations: List[str]      # e.g. ["elbow_flare", "incomplete_range_of_motion"]
    reps_completed: int
    sets_completed: int
    health_snapshot: Dict[BodyRegion, int]  # user's health at time of exercise


@dataclass
class UserFeedback:
    id: str
    exercise_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    rating: int               # 1–5 stars
    comment: Optional[str] = None


@dataclass
class CommunityRecord:
    """Aggregated feedback entry from the community, used for collaborative filtering."""
    id: str
    exercise_id: str
    health_ratings: Dict[BodyRegion, int]   # health profile of the contributor
    user_rating: float                       # normalised 0–1
    timestamp: datetime


@dataclass
class ExerciseRecommendation:
    exercise: CatalogExercise
    score: float                   # 0–1 final recommendation score
    reason: str
    scenario: HealthScenario
    personal_score: float
    community_score: Optional[float]
    feedback_score: Optional[float]


@dataclass
class UserSession:
    id: str
    user_id: str
    target_region: BodyRegion
    health_status: HealthStatus
    started_at: datetime = field(default_factory=datetime.now)
    recommendations: List[ExerciseRecommendation] = field(default_factory=list)
    completed_exercises: List[ExercisePerformanceRecord] = field(default_factory=list)
    feedbacks: List[UserFeedback] = field(default_factory=list)
