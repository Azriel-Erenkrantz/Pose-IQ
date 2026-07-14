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
[Auth]               →  RegisterRequest, LoginRequest, AuthToken
[UI screens]         →  DashboardData, SessionScreenData, HistoryScreenData,
                        ProfileSetupScreenData

DB table mapping (one dataclass → one table)
--------------------------------------------
User                   → users
HealthStatus           → health_status
Exercise               → exercises  (seeded, not user-generated)
LiveSessionOutput      → workout_sessions
RepResult              → rep_records  (FK: session_id)
InjuryRisk             → injury_risks
WeightRecommendation   → weight_recommendations
ExerciseRecommendation → recommendations  (optional — can be computed on the fly)
LiveFeedback           → ephemeral, not persisted
ProgressMetrics        → computed view, not a table
AuthToken              → auth_tokens  (or JWT — no table needed)
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


class TrainerPersonality(str, Enum):
    TOUGH      = "tough"       # direct and demanding
    CALM       = "calm"        # gentle and patient
    MOTIVATING = "motivating"  # energetic and encouraging


class Equipment(str, Enum):
    DUMBBELLS        = "dumbbells"
    RESISTANCE_BANDS = "resistance_bands"
    NONE             = "none"


class TargetGoal(str, Enum):
    """High-level training goal selected during onboarding."""
    LEGS       = "legs"
    CARDIO     = "cardio"
    ABS        = "abs"
    ARMS       = "arms"
    FULL_BODY  = "full_body"
    OTHER      = "other"


class ScoreTrend(str, Enum):
    IMPROVING = "improving"
    STABLE    = "stable"
    DECLINING = "declining"


class HealthScenario(str, Enum):
    ALL_HEALTHY                 = "all_healthy"
    TARGET_HEALTHY_ADJACENT_NOT = "target_healthy_adjacent_not"
    TARGET_UNHEALTHY            = "target_unhealthy"


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
    trainer_personality: TrainerPersonality  = TrainerPersonality.MOTIVATING
    target_goals:        List[TargetGoal]    = field(default_factory=list)
    equipment:           List[Equipment]     = field(default_factory=list)
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
    trainer_personality: TrainerPersonality  = TrainerPersonality.MOTIVATING
    target_goals:        List[TargetGoal]    = field(default_factory=list)
    equipment:           List[Equipment]     = field(default_factory=list)
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
    primary_region:     BodyRegion
    body_regions:       List[BodyRegion]
    base_difficulty:    float               # 0–1, inherent difficulty of the movement
    description:        str
    tags:               List[str]           = field(default_factory=list)
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
    duration_seconds: float             = 0.0


@dataclass
class LiveFeedback:
    """
    Real-time signal emitted frame-by-frame during a rep.
    Ephemeral — shown on screen, never stored.
    """
    timestamp:         datetime
    form_score:        float             # 0–100, instantaneous
    active_violations: List[str]        # joints currently wrong
    coaching_message:  str              # e.g. "straighten your back"


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


# ── Injury and safety ──────────────────────────────────────────────────────────

@dataclass
class InjuryRisk:
    """
    Risk assessment produced by the injury predictor component.
    Based on compensation patterns, asymmetry, and joint stress over time.
    """
    user_id:                 str
    risk_areas:              List[str]   # e.g. ["knee_right", "lower_back"]
    compensation_patterns:   List[str]   # e.g. ["hip_shift_left"]
    asymmetry_score:         float       # 0–1 (0 = symmetric, 1 = severe)
    overall_risk:            float       # 0–1
    recommendation:          str
    assessed_at:             datetime    = field(default_factory=datetime.now)

    @property
    def has_warning(self) -> bool:
        return self.overall_risk >= 0.4


# ── Weight recommendation ──────────────────────────────────────────────────────

@dataclass
class WeightRecommendation:
    """
    Dynamic load recommendation for one exercise, based on performance analysis.
    """
    exercise_id:            str
    exercise_name:          str
    recommended_weight_kg:  float
    reasoning:              str
    confidence:             float        # 0–1


# ── Recommendation ─────────────────────────────────────────────────────────────

@dataclass
class ExerciseRecommendation:
    """One ranked recommendation produced by the recommendation engine."""
    exercise:        Exercise
    score:           float               # 0–1 final blended score
    reason:          str
    scenario:        HealthScenario
    personal_score:  float
    community_score: Optional[float]
    feedback_score:  Optional[float]


@dataclass
class ExercisePerformanceRecord:
    """
    Output contract for the performance-monitoring ML model.
    Records one completed exercise set; feeds the recommendation engine.
    """
    id:                 str
    exercise_id:        str
    user_id:            str
    session_id:         str
    timestamp:          datetime
    difficulty_score:   float           # 0–1
    form_quality_score: float           # 0–1
    completion_rate:    float           # 0–1
    violations:         List[str]
    reps_completed:     int
    sets_completed:     int
    health_snapshot:    Dict[BodyRegion, int]


@dataclass
class UserFeedback:
    """Explicit 1–5 star rating the user gives after an exercise set."""
    id:          str
    exercise_id: str
    user_id:     str
    session_id:  str
    timestamp:   datetime
    rating:      int
    comment:     Optional[str] = None


@dataclass
class CommunityRecord:
    """Aggregated feedback from one community user, used for collaborative filtering."""
    id:             str
    exercise_id:    str
    health_ratings: Dict[BodyRegion, int]
    user_rating:    float               # normalised 0–1
    timestamp:      datetime


@dataclass
class RecommendationSession:
    """In-memory session used by the recommendation simulation layer."""
    id:                   str
    user_id:              str
    target_region:        BodyRegion
    health_status:        HealthStatus
    started_at:           datetime                     = field(default_factory=datetime.now)
    recommendations:      List[ExerciseRecommendation] = field(default_factory=list)
    completed_exercises:  List[ExercisePerformanceRecord] = field(default_factory=list)
    feedbacks:            List[UserFeedback]           = field(default_factory=list)


@dataclass
class RecommendationInput:
    """
    Everything the recommendation engine needs from the user layer.
    Built by the API; passed to recommendation/bridge.recommend_for_user().
    """
    fitness_level:   FitnessLevel
    target_goals:    List[TargetGoal]
    equipment:       List[Equipment]
    limited_joints:  List[str]          # already mapped — not raw limitation strings
    health_status:   HealthStatus
    session_history: List[LiveSessionOutput]


# ── UI screen data contracts ───────────────────────────────────────────────────

@dataclass
class ProfileSetupScreenData:
    """
    Static options the onboarding screen needs to render its pickers.
    Returned by the API; mobile app renders these as selectable lists.
    """
    fitness_levels:       List[FitnessLevel]
    trainer_personalities: List[TrainerPersonality]
    target_goals:         List[TargetGoal]
    equipment_options:    List[Equipment]
    limitation_options:   List[str]      # free-text body part limitations


@dataclass
class DashboardData:
    """Everything the main dashboard screen needs to render."""
    user:             User
    health_status:    HealthStatus
    recommendations:  List[ExerciseRecommendation]
    recent_sessions:  List[LiveSessionOutput]
    progress_summary: List[ProgressMetrics]
    injury_risk:      Optional[InjuryRisk]  = None
    weight_recommendations: List[WeightRecommendation] = field(default_factory=list)


@dataclass
class SessionScreenData:
    """
    Everything the live exercise screen needs.
    live_feedback is None between reps; populated frame-by-frame during a rep.
    """
    user:           User
    exercise:       Exercise
    completed_reps: List[RepResult]
    live_feedback:  Optional[LiveFeedback]  = None
    session_id:     str                     = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class HistoryScreenData:
    """Everything the history / progress screen needs."""
    user:     User
    sessions: List[LiveSessionOutput]
    metrics:  Dict[str, ProgressMetrics]    # keyed by exercise_id
