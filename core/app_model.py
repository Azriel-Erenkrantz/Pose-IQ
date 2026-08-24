"""
Master data model for Pose-IQ.

Single source of truth for every data contract that crosses a boundary
between app parts, the UI, or the database.

No imports from anywhere else in this project — only stdlib.
All other modules depend on this file; this file depends on nothing.

Boundary map
------------
[Camera + ML model]  →  LiveSessionOutput, RepResult
[Session layer]      →  LiveSessionOutput  →  [History / DB]
[Recommendation]     →  ExerciseRecommendation  →  [UI]
[User]               →  User, HealthStatus  →  [All parts]
[Auth]               →  RegisterRequest, LoginRequest, AuthToken
[UI screens]         →  DashboardData, ProfileSetupScreenData

DB table mapping (one dataclass → one table)
--------------------------------------------
User                   → users
HealthStatus           → health_status
Exercise               → exercises  (seeded, not user-generated)
LiveSessionOutput      → workout_sessions
RepResult              → rep_records  (FK: session_id)
ExerciseRecommendation → recommendations  (optional — can be computed on the fly)
ProgressMetrics        → computed view, not a table
AuthToken              → auth_tokens  (or JWT — no table needed)
"""
from __future__ import annotations

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


class Equipment(str, Enum):
    DUMBBELLS        = "dumbbells"
    RESISTANCE_BANDS = "resistance_bands"
    NONE             = "none"


class ScoreTrend(str, Enum):
    IMPROVING = "improving"
    STABLE    = "stable"
    DECLINING = "declining"


# ── User ──────────────────────────────────────────────────────────────────────

# Maps user-reported limitation labels to the joint names the evaluation pipeline checks.
# Lives here so any module that receives a User can compute limited_joints without
# importing from the user folder.
LIMITATION_JOINT_MAP: Dict[str, List[str]] = {
    'right_knee':    ['right_knee'],
    'left_knee':     ['left_knee'],
    'lower_back':    ['spine'],
    'right_shoulder':['right_arm_body'],
    'left_shoulder': ['left_arm_body'],
    'right_elbow':   ['right_elbow'],
    'left_elbow':    ['left_elbow'],
}


@dataclass
class User:
    """
    Who is using the app.
    Maps to core/user/UserProfile — will consolidate on unisolation.
    """
    user_id:             str
    name:                str
    email:               str
    fitness_level:       FitnessLevel        = FitnessLevel.INTERMEDIATE
    limitations:         List[str]           = field(default_factory=list)
    created_at:          datetime            = field(default_factory=datetime.now)

    @property
    def threshold_modifier(self) -> float:
        """How strictly the evaluation pipeline checks joint angles.
        Beginners get wider acceptable ranges; advanced users get tighter ones."""
        return {
            FitnessLevel.BEGINNER:     1.30,
            FitnessLevel.INTERMEDIATE: 1.00,
            FitnessLevel.ADVANCED:     0.85,
        }[self.fitness_level]

    @property
    def limited_joints(self) -> List[str]:
        """Joint names the pipeline should evaluate with extra leniency.
        Derived from the user's reported physical limitations."""
        joints: List[str] = []
        for limitation in self.limitations:
            joints.extend(LIMITATION_JOINT_MAP.get(limitation, []))
        return joints


# ── Auth ──────────────────────────────────────────────────────────────────────

@dataclass
class RegisterRequest:
    name:                str
    email:               str
    password:            str                 # plaintext — hashed at service layer
    fitness_level:       FitnessLevel        = FitnessLevel.INTERMEDIATE
    limitations:         List[str]           = field(default_factory=list)


@dataclass
class LoginRequest:
    email:    str
    password: str


@dataclass
class AuthToken:
    user_id:    str
    token:      str
    expires_at: datetime


# ── Health ─────────────────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    """
    User's body condition at a point in time.
    Maps to core/recommendation/models.HealthStatus — will consolidate on unisolation.
    """
    user_id:     str
    ratings:     Dict[BodyRegion, int]    # 1 = poor, 5 = fully healthy
    recorded_at: datetime                 = field(default_factory=datetime.now)

    def get(self, region: BodyRegion) -> int:
        return self.ratings.get(region, 3)


# ── Exercise catalog ───────────────────────────────────────────────────────────

@dataclass
class Exercise:
    """
    Unified exercise definition used across all parts.
    Maps to core/recommendation/models.CatalogExercise — will consolidate on unisolation.
    """
    exercise_id:        str
    name:               str
    description:        str
    equipment_required: List[Equipment]     = field(default_factory=list)


# ── Live session (boundary with the ML model) ─────────────────────────────────

@dataclass
class RepResult:
    """
    Output of one completed rep.
    Produced by the ML model; consumed by session layer, history, and UI.
    """
    rep_number:       int
    form_score:       float              # 0–100
    error_joints:     List[str]         # joints that violated form rules this rep


@dataclass
class LiveSessionOutput:
    """
    Contract at the boundary with the ML model + camera.

    The ML model fills this when ready.
    Until then, FakeSessionGenerator (in core/session/) produces it.
    """
    session_id:       str
    exercise_id:      str
    exercise_name:    str
    user_id:          str
    date:             datetime
    reps:             List[RepResult]
    duration_seconds: float
    overall_score:    float              # 0–100, avg form across all reps
    weight_kg:        Optional[float] = None   # load used; None = not logged, 0 = bodyweight

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def weak_joints(self) -> List[str]:
        return list({j for rep in self.reps for j in rep.error_joints})


# ── History and progress ───────────────────────────────────────────────────────

@dataclass
class ProgressMetrics:
    """
    Aggregated history for one exercise.
    Computed from LiveSessionOutput records — not stored as its own DB table.
    """
    exercise_id:       str
    exercise_name:     str
    total_sessions:    int
    total_reps:        int
    avg_score_recent:  float
    score_trend:       ScoreTrend
    weak_joints:       List[Tuple[str, int]]  # (joint_name, error_count)


# ── Recommendation ─────────────────────────────────────────────────────────────

@dataclass
class ExerciseRecommendation:
    """
    One ranked recommendation produced by the rating-based ranker.

    `reason` is a fixed English sentence (kept for backward compatibility).
    `reason_code` + `reason_params` let a client render the same message in
    any supported language — see i18n.tsx `formatReason`.
    """
    exercise:      Exercise
    score:         float               # 0–1 predicted rating, normalised
    reason:        str
    reason_code:   str = ""
    reason_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class Rating:
    """A user's own 1–5 star rating of an exercise. One per (user, exercise)."""
    user_id:     str
    exercise_id: str
    rating:      int
    rated_at:    datetime = field(default_factory=datetime.now)


# ── UI screen data contracts ───────────────────────────────────────────────────

@dataclass
class ProfileSetupScreenData:
    """
    Static options the onboarding screen needs to render its pickers.
    Returned by the API; the client renders these as selectable lists.
    """
    fitness_levels:       List[FitnessLevel]
    limitation_options:   List[str]      # free-text body part limitations


@dataclass
class DashboardData:
    """Everything the main dashboard screen needs to render."""
    user:             User
    health_status:    HealthStatus
    recommendations:  List[ExerciseRecommendation]
    recent_sessions:  List[LiveSessionOutput]
    progress_summary: List[ProgressMetrics]
