from typing import Dict, List, Optional, Tuple
from user.workout_history import WorkoutHistory
from user.user_profile import UserProfile

# Which joints indicate weakness in which exercises
JOINT_TO_CORRECTIVE = {
    'right_knee': ['squat', 'lunge'],
    'left_knee': ['squat', 'lunge'],
    'spine': ['plank', 'squat'],
    'right_elbow': ['bicep_curl', 'shoulder_press'],
    'left_elbow': ['bicep_curl', 'shoulder_press'],
    'right_arm_body': ['shoulder_press'],
    'left_arm_body': ['shoulder_press'],
    'legs_spread': ['squat', 'lunge'],
}

# Joints that are problematic for each limitation
LIMITATION_CONFLICTS = {
    'right_knee': ['squat', 'lunge'],
    'left_knee': ['squat', 'lunge'],
    'lower_back': ['squat', 'lunge', 'plank'],
    'right_shoulder': ['shoulder_press'],
    'left_shoulder': ['shoulder_press'],
    'right_elbow': ['bicep_curl', 'shoulder_press'],
    'left_elbow': ['bicep_curl', 'shoulder_press'],
}

BASE_TARGET_REPS = {
    'beginner': 8,
    'intermediate': 12,
    'advanced': 16,
}


class WorkoutPlan:
    def __init__(self, profile: UserProfile, history: WorkoutHistory):
        self.profile = profile
        self.history = history

    def recommend_exercises(self, available: List[str]) -> List[dict]:
        """Score and rank exercises for this session."""
        conflicting = self._conflicting_exercises()
        weak_joints = [j for j, _ in self.history.get_weak_muscle_groups()]
        recent_ids = [s.exercise_id for s in self.history.get_sessions(limit=3)]

        ranked = []
        for ex_id in available:
            score = 0
            reasons = []

            if ex_id in conflicting:
                continue

            if ex_id in self.profile.preferred_exercises:
                score += 3
                reasons.append("preferred")

            # Recommend exercises that target weak joints
            for joint in weak_joints:
                if ex_id in JOINT_TO_CORRECTIVE.get(joint, []):
                    score += 2
                    reasons.append(f"targets weak area ({joint})")
                    break

            # Recommend exercises with recent poor performance for practice
            for s in self.history.get_sessions(exercise_id=ex_id, limit=3):
                if s.overall_score < 70:
                    score += 2
                    reasons.append("needs practice")
                    break

            # Slight boost for variety (not done recently)
            if ex_id not in recent_ids:
                score += 1
                reasons.append("variety")

            ranked.append({'exercise_id': ex_id, 'score': score, 'reasons': reasons})

        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

    def get_difficulty_settings(self, exercise_id: str) -> dict:
        """Return suggested target reps and angle modifier based on history."""
        base_reps = BASE_TARGET_REPS.get(self.profile.fitness_level, 12)
        metrics = self.history.get_progress_metrics(exercise_id)

        reps = base_reps
        angle_hint = "standard"

        if metrics:
            avg_score = metrics.get('avg_score_recent', 0)
            trend = metrics.get('score_trend', 'neutral')

            if avg_score >= 85 and trend == 'improving':
                reps = base_reps + 3
                angle_hint = "stricter"
            elif avg_score < 60 or trend == 'declining':
                reps = max(base_reps - 3, 5)
                angle_hint = "relaxed"

        return {
            'target_reps': reps,
            'angle_hint': angle_hint,
            'fitness_level': self.profile.fitness_level,
        }

    def get_corrective_recommendations(self, available: List[str]) -> List[Tuple[str, str]]:
        """Return (exercise_id, reason) pairs targeting the user's weak joints."""
        weak_joints = [j for j, count in self.history.get_weak_muscle_groups()]
        conflicting = self._conflicting_exercises()
        seen = set()
        result = []

        for joint in weak_joints:
            for ex_id in JOINT_TO_CORRECTIVE.get(joint, []):
                if ex_id in available and ex_id not in conflicting and ex_id not in seen:
                    result.append((ex_id, f"strengthen {joint.replace('_', ' ')}"))
                    seen.add(ex_id)

        return result[:3]

    def _conflicting_exercises(self) -> set:
        conflicting = set()
        for limitation in self.profile.limitations:
            conflicting.update(LIMITATION_CONFLICTS.get(limitation, []))
        return conflicting

    def print_plan(self, available: List[str]):
        recommendations = self.recommend_exercises(available)
        difficulty = {}
        for rec in recommendations[:3]:
            difficulty[rec['exercise_id']] = self.get_difficulty_settings(rec['exercise_id'])

        corrective = self.get_corrective_recommendations(available)

        print("\n" + "=" * 50)
        print(f"  WORKOUT PLAN — {self.profile.name}")
        print(f"  Level: {self.profile.fitness_level}")
        print("=" * 50)

        print("\n  Recommended for today:")
        for i, rec in enumerate(recommendations[:3], 1):
            ex_id = rec['exercise_id']
            diff = difficulty.get(ex_id, {})
            reasons = ', '.join(rec['reasons']) if rec['reasons'] else 'general'
            print(f"    {i}. {ex_id}  →  {diff.get('target_reps', '?')} reps  ({reasons})")

        if corrective:
            print("\n  Focus on (based on your weak areas):")
            for ex_id, reason in corrective:
                print(f"    • {ex_id}: {reason}")

        print()
