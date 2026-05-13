from typing import Dict, List, Optional
from exercise_model import Exercise, Phase, AngleRange
from user_profile import UserProfile


class PostureRules:
    FRAMES_TO_ALERT = 8

    def __init__(self, exercise: Exercise, profile: Optional[UserProfile] = None):
        self.exercise = exercise
        self.corrections = exercise.corrections
        self.profile = profile
        self.modifier = profile.threshold_modifier if profile else 1.0
        self.limited_joints = profile.limited_joints if profile else []
        self.violation_counters: Dict[str, int] = {}

    def _adjust_range(self, joint: str, angle_range: AngleRange) -> AngleRange:
        if self.modifier == 1.0 and joint not in self.limited_joints:
            return angle_range

        mid = (angle_range.min + angle_range.max) / 2
        half_span = (angle_range.max - angle_range.min) / 2

        adjusted_half = half_span * self.modifier
        if joint in self.limited_joints:
            adjusted_half *= 1.4

        return AngleRange(
            min=round(mid - adjusted_half, 1),
            max=round(mid + adjusted_half, 1)
        )

    def analyze(self, angles: Dict[str, float], phase: Phase) -> List[dict]:
        issues = []
        if not angles or not phase:
            return issues

        active_joints = set()

        for joint, angle_range in phase.angles.items():
            if joint not in angles:
                continue

            active_joints.add(joint)
            value = angles[joint]
            adjusted = self._adjust_range(joint, angle_range)

            if adjusted.contains(value):
                self.violation_counters[joint] = 0
                continue

            self.violation_counters[joint] = self.violation_counters.get(joint, 0) + 1

            if self.violation_counters[joint] < self.FRAMES_TO_ALERT:
                continue

            direction = "too_low" if value < adjusted.min else "too_high"
            correction = self.corrections.get(joint, {})
            message = correction.get(direction, f"{joint}: {direction} ({value:.0f}, expected {adjusted.min:.0f}-{adjusted.max:.0f})")
            severity = correction.get("severity", "medium")

            if joint in self.limited_joints:
                severity = "low"
                message = f"[Adapted] {message}"

            issues.append({
                'joint': joint,
                'severity': severity,
                'message': message,
                'value': round(value),
                'expected_min': adjusted.min,
                'expected_max': adjusted.max
            })

        for joint in list(self.violation_counters.keys()):
            if joint not in active_joints:
                self.violation_counters[joint] = 0

        return issues

    def reset(self):
        self.violation_counters.clear()
