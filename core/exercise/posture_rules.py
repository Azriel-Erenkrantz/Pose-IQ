from typing import Dict, List
from core.exercise.exercise_model import Exercise, AngleRange


class PostureRules:
    FRAMES_TO_ALERT = 8
    FRAMES_TO_MISSING = 5

    def __init__(self, exercise: Exercise, threshold_modifier: float = 1.0, limited_joints: List[str] = []):
        self.exercise = exercise
        self.modifier = threshold_modifier
        self.limited_joints = limited_joints
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
            max=round(mid + adjusted_half, 1),
            corrections=angle_range.corrections,
        )

    def analyze(self, angles: Dict[str, float], active_rules: Dict[str, AngleRange]) -> List[dict]:
        """
        Evaluate current joint angles against active rules (global + phase merged).
        Returns a list of violation dicts, one per persistently wrong joint.
        """
        issues = []
        if not active_rules:
            return issues

        for joint, angle_range in active_rules.items():
            if joint not in angles:
                self.violation_counters[joint] = self.violation_counters.get(joint, 0) + 1
                if self.violation_counters[joint] >= self.FRAMES_TO_MISSING:
                    issues.append({
                        'joint': joint,
                        'severity': 'high',
                        'message': f"Can't see your {joint.replace('_', ' ')} — make sure it's in frame",
                        'direction': 'missing',
                        'value': None,
                        'expected_min': angle_range.min,
                        'expected_max': angle_range.max,
                    })
                continue

            value = angles[joint]
            adjusted = self._adjust_range(joint, angle_range)

            if adjusted.contains(value):
                self.violation_counters[joint] = 0
                continue

            self.violation_counters[joint] = self.violation_counters.get(joint, 0) + 1
            if self.violation_counters[joint] < self.FRAMES_TO_ALERT:
                continue

            message, severity = adjusted.correction_message(value)

            if joint in self.limited_joints:
                severity = 'low'
                message = f"[Adapted] {message}"

            issues.append({
                'joint': joint,
                'severity': severity,
                'message': message,
                'direction': 'too_low' if value < adjusted.min else 'too_high',
                'value': round(value),
                'expected_min': adjusted.min,
                'expected_max': adjusted.max,
            })

        for joint in list(self.violation_counters.keys()):
            if joint not in active_rules:
                self.violation_counters[joint] = 0

        return issues

    def reset(self):
        self.violation_counters.clear()
