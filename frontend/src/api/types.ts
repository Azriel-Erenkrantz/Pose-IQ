// TypeScript contracts mirroring core/app_model.py
// Keep in sync when app_model.py changes.

export type FitnessLevel = 'beginner' | 'intermediate' | 'advanced';
export type Equipment = 'dumbbells' | 'resistance_bands' | 'none';
export type ScoreTrend = 'improving' | 'stable' | 'declining';
export type BodyRegion = 'upper' | 'core' | 'lower';

export interface User {
  user_id: string;
  name: string;
  email: string;
  fitness_level: FitnessLevel;
  limitations: string[];
  created_at: string;
}

export interface HealthStatus {
  user_id: string;
  ratings: Record<BodyRegion, number>;
  recorded_at: string;
}

export interface RepResult {
  rep_number: number;
  form_score: number;
  error_joints: string[];
}

export interface LiveSessionOutput {
  session_id: string;
  exercise_id: string;
  exercise_name: string;
  user_id: string;
  date: string;
  reps: RepResult[];
  duration_seconds: number;
  overall_score: number;
  weight_kg: number | null;
}

export interface WeightRecommendation {
  exercise_id: string;
  exercise_name: string;
  recommended_weight_kg: number;
  reasoning: string;
  confidence: number;
  reasoning_code: string;
  reasoning_params: Record<string, number>;
}

export interface ProgressMetrics {
  exercise_id: string;
  exercise_name: string;
  total_sessions: number;
  total_reps: number;
  avg_score_recent: number;
  score_trend: ScoreTrend;
  weak_joints: [string, number][];
}

export interface ExerciseInfo {
  exercise_id: string;
  name: string;
  description: string;
  equipment_required: Equipment[];
}

export interface ExerciseRecommendation {
  exercise: ExerciseInfo;
  score: number;
  reason: string;
  reason_code: string;
  reason_params: Record<string, number>;
}

// exercise_id -> rating (1-5); GET /api/user/<id>/ratings
export type UserRatings = Record<string, number>;

export interface DashboardData {
  user: User;
  health_status: HealthStatus;
  recommendations: ExerciseRecommendation[];
  recent_sessions: LiveSessionOutput[];
  progress_summary: ProgressMetrics[];
  weight_recommendations: WeightRecommendation[];
}

export interface AuthToken {
  user_id: string;
  token: string;
  expires_at: string;
}

// ── Exercise model for the in-browser state machine (GET /api/exercises) ──

export interface AngleRangeDef {
  min: number;
  max: number;
  corrections: Record<string, string>;   // too_low / too_high / severity
  mean: number | null;
  std: number | null;
}

export interface PhaseDef {
  name: string;
  order: number;
  angles: Record<string, AngleRangeDef>;
  instruction: string;
  is_initial: boolean;
  diagnostic_joints: string[];
  motion_direction: 'stable' | 'increasing' | 'decreasing';
}

export interface ExerciseDef {
  id: string;
  name: string;
  description: string;
  muscle_groups: string[];
  phases: PhaseDef[];
  primary_joints: string[];
  mandatory_start_joints: string[];
  global_constraints: Record<string, AngleRangeDef>;
  ready: boolean;   // trainer has written angle ranges — state machine can run
}

export interface RepPayload {
  rep_number: number;
  error_joints: string[];
  form_score: number;
}

export interface SavedSession {
  session_id: string;
  total_reps: number;
  overall_score: number;
  weight_kg: number | null;
}

export interface ProfileSetupOptions {
  fitness_levels: FitnessLevel[];
  limitation_options: string[];
}
