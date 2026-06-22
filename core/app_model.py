"""
Master data model for Pose-IQ.

Single source of truth for every data contract that crosses a boundary
between app parts, the UI, or the database.

No imports from anywhere else in this project — only stdlib.
All other modules depend on this file; this file depends on nothing.

Boundary map
------------
[Camera + ML model]  →  LiveSessionOutput, RepResult, LiveFeedback
[Session layer]      →  LiveSessionOutput  →  [History / DB]
[Recommendation]     →  ExerciseRecommendation  →  [UI]
[User]               →  User, HealthStatus  →  [All parts]
[UI screens]         →  DashboardData, SessionScreenData, HistoryScreenData

Relationship to existing module models
---------------------------------------
These types are the FUTURE unified state. Existing module-internal types
(recommendation/models.py, user/user_profile.py, user/workout_history.py)
remain in place until each module is unisolated and updated to import
from here instead.

DB table mapping (one dataclass → one table)
--------------------------------------------
User                → users
HealthStatus        → health_status
Exercise            → exercises  (or seeded from catalog, not user-generated)
LiveSessionOutput   → workout_sessions
RepResult           → rep_records  (FK: session_id)
ExerciseRecommendation → recommendations  (optional — can be computed on the fly)
LiveFeedback        → ephemeral, not persisted
ProgressMetrics     → computed view, not a table
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Enums ─────────────────────────────────────────────────────────────────────

class BodyRegion(str, Enum):
    UPPER = "upper"
    CORE  = "core"
    LOWER = "lower"


class FitnessLevel(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"


class ScoreTrend(str, Enum):
    IMPROVING = "improving"
    STABLE    = "stable"
    DECLINING = "declining"


# ── User ──────────────────────────────────────────────────────────────────────

@dataclass
class User:
    """
    Who is using the app.
    Maps to core/user/UserProfile — will consolidate on unisolation.
    """
    user_id: str
    name: str
    fitness_level: FitnessLevel          = FitnessLevel.INTERMEDIATE
    limitations: List[str]               = field(default_factory=list)
    preferred_exercises: List[str]       = field(default_factory=list)
    coach_style: str                     = "motivator"
    created_at: datetime                 = field(default_factory=datetime.now)

    @property
    def threshold_modifier(self) -> float:
        return {
            FitnessLevel.BEGINNER:     1.30,
            FitnessLevel.INTERMEDIATE: 1.00,
            FitnessLevel.ADVANCED:     0.85,
        }[self.fitness_level]


# ── Health ─────────────────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    """
    User's body condition at a point in time.
    Maps to core/recommendation/models.HealthStatus — will consolidate on unisolation.
    """
    user_id: str
    ratings: Dict[BodyRegion, int]       # 1 = poor, 5 = fully healthy
    recorded_at: datetime                = field(default_factory=datetime.now)

    def get(self, region: BodyRegion) -> int:
        return self.ratings.get(region, 3)


# ── Exercise catalog ───────────────────────────────────────────────────────────

@dataclass
class Exercise:
    """
    Unified exercise definition used across all parts.
    Maps to core/recommendation/models.CatalogExercise — will consolidate on unisolation.
    """
    exercise_id: str
    name: str
    primary_region: BodyRegion
    body_regions: List[BodyRegion]
    base_difficulty: float               # 0–1, inherent difficulty of the movement
    description: str
    tags: List[str]                      = field(default_factory=list)


# ── Live session (boundary with the ML model) ─────────────────────────────────

@dataclass
class RepResult:
    """
    Output of one completed rep.
    Produced by the ML model; consumed by session layer, history, and UI.
    """
    rep_number: int
    form_score: float                    # 0–100
    error_joints: List[str]             # joints that violated form rules this rep
    duration_seconds: float              = 0.0


@dataclass
class LiveFeedback:
    """
    Real-time signal emitted frame-by-frame during a rep.
    Ephemeral — shown on screen, never stored.
    This is the direct output of the ML model during exercise.
    """
    timestamp: datetime
    form_score: float                    # 0–100, instantaneous
    active_violations: List[str]        # joints currently wrong
    coaching_message: str               # "straighten your back", "slow down", etc.


@dataclass
class LiveSessionOutput:
    """
    Contract at the boundary with the ML model + camera.

    The ML model fills this when ready.
    Until then, FakeSessionGenerator (in core/session/) produces it.

    This is the single type that:
      - The session layer receives from the ML model
      - The history layer persists to the DB
      - The recommendation layer reads (via bridge) to update scores
      - The UI displays on the session summary screen
    """
    session_id: str
    exercise_id: str
    exercise_name: str
    user_id: str
    date: datetime
    reps: List[RepResult]
    duration_seconds: float
    overall_score: float                 # 0–100, avg form across all reps

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def weak_joints(self) -> List[str]:
        """All joints that caused errors across any rep, deduplicated."""
        return list({j for rep in self.reps for j in rep.error_joints})


# ── History and progress ───────────────────────────────────────────────────────

@dataclass
class ProgressMetrics:
    """
    Aggregated history for one exercise.
    Computed from LiveSessionOutput records — not stored as its own DB table.
    What the progress / history UI screen displays.
    """
    exercise_id: str
    exercise_name: str
    total_sessions: int
    total_reps: int
    avg_score_recent: float
    score_trend: ScoreTrend
    weak_joints: List[Tuple[str, int]]   # (joint_name, error_count)


# ── Recommendation ─────────────────────────────────────────────────────────────

@dataclass
class ExerciseRecommendation:
    """
    One ranked recommendation produced by the recommendation engine.
    Maps to core/recommendation/models.ExerciseRecommendation — will consolidate on unisolation.
    """
    exercise: Exercise
    score: float                         # 0–1 final blended score
    reason: str
    scenario: str                        # health scenario name
    personal_score: float
    community_score: Optional[float]
    feedback_score: Optional[float]


# ── UI screen data contracts ───────────────────────────────────────────────────

@dataclass
class DashboardData:
    """
    Everything the main dashboard screen needs to render.
    Assembled by the app layer from all isolated parts.
    """
    user: User
    health_status: HealthStatus
    recommendations: List[ExerciseRecommendation]
    recent_sessions: List[LiveSessionOutput]
    progress_summary: List[ProgressMetrics]


@dataclass
class SessionScreenData:
    """
    Everything the live exercise screen needs.
    live_feedback is None between reps; populated frame-by-frame during a rep.
    """
    user: User
    exercise: Exercise
    completed_reps: List[RepResult]
    live_feedback: Optional[LiveFeedback]  = None
    session_id: str                        = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class HistoryScreenData:
    """
    Everything the history / progress screen needs.
    """
    user: User
    sessions: List[LiveSessionOutput]
    metrics: Dict[str, ProgressMetrics]   # keyed by exercise_id
