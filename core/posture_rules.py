from typing import Dict, List
from exercise_model import Exercise, Phase


class PostureRules:
    FRAMES_TO_ALERT = 8

    def __init__(self, exercise: Exercise):
        self.exercise = exercise
        self.corrections = exercise.corrections if hasattr(exercise, 'corrections') else {}
        self.violation_counters: Dict[str, int] = {}

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

            if angle_range.contains(value):
                self.violation_counters[joint] = 0
                continue

            self.violation_counters[joint] = self.violation_counters.get(joint, 0) + 1

            if self.violation_counters[joint] < self.FRAMES_TO_ALERT:
                continue

            direction = "too_low" if value < angle_range.min else "too_high"
            correction = self.corrections.get(joint, {})
            message = correction.get(direction, f"{joint}: {direction} ({value:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})")
            severity = correction.get("severity", "medium")

            issues.append({
                'joint': joint,
                'severity': severity,
                'message': message,
                'value': round(value),
                'expected_min': angle_range.min,
                'expected_max': angle_range.max
            })

        for joint in list(self.violation_counters.keys()):
            if joint not in active_joints:
                self.violation_counters[joint] = 0

        return issues

    def reset(self):
        self.violation_counters.clear()
