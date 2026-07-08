"""
Violation detector for Model 1.

Given the detected phase and current joint angles, checks every joint in the
phase's angle_ranges plus the exercise's global_constraints. Returns a list of
violation dicts with the specific correction message for that phase + body part.

Violations are debounced: a joint must be out of range for FRAMES_TO_ALERT
consecutive frames before it's reported. This avoids noisy alerts during
transitions between phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.exercise.exercise_model import AngleRange, Exercise, Phase

FRAMES_TO_ALERT = 8    # frames a joint must be wrong before reporting
FRAMES_TO_MISSING = 5  # frames a joint must be absent before reporting


@dataclass
class Violation:
    joint: str
    phase: str
    direction: str        # 'too_low' | 'too_high' | 'missing'
    message: str
    severity: str         # 'high' | 'medium' | 'low'
    value: Optional[float]
    expected_min: float
    expected_max: float

    def to_dict(self) -> dict:
        return {
            'joint':        self.joint,
            'phase':        self.phase,
            'direction':    self.direction,
            'message':      self.message,
            'severity':     self.severity,
            'value':        self.value,
            'expected_min': self.expected_min,
            'expected_max': self.expected_max,
        }


class ViolationDetector:
    def __init__(self, exercise: Exercise, limited_joints: List[str] = None):
        self.exercise = exercise
        self.limited_joints: List[str] = limited_joints or []
        self._counters: Dict[str, int] = {}  # joint → consecutive bad frames

    def analyze(self, phase: Phase, angles: Dict[str, float]) -> List[Violation]:
        """
        Check angles against the current phase rules + global constraints.
        Returns only violations that have persisted long enough to report.
        """
        # Merge global constraints with phase-specific rules.
        # Phase rules take priority if the same joint appears in both.
        active: Dict[str, AngleRange] = {**self.exercise.global_constraints, **phase.angles}

        violations: List[Violation] = []

        for joint, ar in active.items():
            value = angles.get(joint)

            if value is None:
                count = self._counters.get(joint, 0) + 1
                self._counters[joint] = count
                if count >= FRAMES_TO_MISSING:
                    violations.append(Violation(
                        joint=joint, phase=phase.name, direction='missing',
                        message=f"Can't see your {joint.replace('_', ' ')} — make sure it's in frame",
                        severity='high', value=None,
                        expected_min=ar.min, expected_max=ar.max,
                    ))
                continue

            if ar.contains(value):
                self._counters[joint] = 0
                continue

            count = self._counters.get(joint, 0) + 1
            self._counters[joint] = count
            if count < FRAMES_TO_ALERT:
                continue

            direction = 'too_low' if value < ar.min else 'too_high'
            message, severity = ar.correction_message(value)

            if joint in self.limited_joints:
                severity = 'low'
                message = f"[Adapted] {message}"

            violations.append(Violation(
                joint=joint, phase=phase.name, direction=direction,
                message=message, severity=severity,
                value=round(value, 1),
                expected_min=ar.min, expected_max=ar.max,
            ))

        # Clear counters for joints no longer in the active rule set
        for joint in list(self._counters):
            if joint not in active:
                self._counters[joint] = 0

        return violations

    def reset(self):
        self._counters.clear()
