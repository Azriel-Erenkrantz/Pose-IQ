from typing import Dict, List, Optional
from exercise.exercise_model import Exercise, AngleRange
from user.user_profile import UserProfile


class PostureRules:
    FRAMES_TO_ALERT = 8
    # TODO 2: Grace period for occlusions before alerting that a joint is missing
    FRAMES_TO_MISSING = 5 

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

    def analyze(self, angles: Dict[str, float], active_rules: Dict[str, AngleRange]) -> List[dict]:
        """
        Evaluates the current angles against the active rules.
        TODO 3: Now accepts 'active_rules' (merged global + phase constraints) instead of just phase.
        """
        issues = []
        if not active_rules:
            return issues

        for joint, angle_range in active_rules.items():
            # TODO 2: Handle extended occlusion (missing joints during exercise)
            if joint not in angles:
                self.violation_counters[joint] = self.violation_counters.get(joint, 0) + 1
                if self.violation_counters[joint] >= self.FRAMES_TO_MISSING:
                    issues.append({
                        'joint': joint,
                        'severity': 'high',
                        'message': f"Lost track of joint {joint}. Please make sure it is visible.",
                        'direction': 'missing',
                        'value': None,
                        'expected_min': angle_range.min,
                        'expected_max': angle_range.max
                    })
                continue

            value = angles[joint]
            adjusted = self._adjust_range(joint, angle_range)

            # Reset counter if the joint is within the valid range
            if adjusted.contains(value):
                self.violation_counters[joint] = 0
                continue

            # Increment counter for incorrect angle
            self.violation_counters[joint] = self.violation_counters.get(joint, 0) + 1

            # Wait until the error persists for enough frames before alerting
            if self.violation_counters[joint] < self.FRAMES_TO_ALERT:
                continue

            direction = "too_low" if value < adjusted.min else "too_high"
            correction = self.corrections.get(joint, {})
            message = correction.get(direction, f"{joint}: {direction} ({value:.0f}, expected {adjusted.min:.0f}-{adjusted.max:.0f})")
            severity = correction.get("severity", "medium")

            # Apply leniency for joints with known physical limitations
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

        # Clean up counters for joints that are no longer in the active rules
        for joint in list(self.violation_counters.keys()):
            if joint not in active_rules:
                self.violation_counters[joint] = 0

        return issues

    def reset(self):
        self.violation_counters.clear()