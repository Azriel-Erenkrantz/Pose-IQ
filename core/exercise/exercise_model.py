import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AngleRange:
    min: float
    max: float

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max


@dataclass
class Phase:
    name: str
    order: int
    angles: Dict[str, AngleRange]


@dataclass
class Exercise:
    id: str
    name: str
    description: str
    muscle_groups: List[str]
    phases: List[Phase]
    corrections: Dict[str, dict]

    def get_phase(self, name: str) -> Optional[Phase]:
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def get_phase_by_order(self, order: int) -> Optional[Phase]:
        for phase in self.phases:
            if phase.order == order:
                return phase
        return None


class ExerciseModel:
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'exercises.json')
        self.exercises: Dict[str, Exercise] = {}
        self._load(data_path)

    def _load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for ex_data in data['exercises']:
            phases = []
            for phase_data in ex_data['phases']:
                angles = {
                    joint: AngleRange(min=r['min'], max=r['max'])
                    for joint, r in phase_data['angles'].items()
                }
                phases.append(Phase(name=phase_data['name'], order=phase_data['order'], angles=angles))

            exercise = Exercise(
                id=ex_data['id'],
                name=ex_data['name'],
                description=ex_data['description'],
                muscle_groups=ex_data['muscle_groups'],
                phases=phases,
                corrections=ex_data.get('corrections', {})
            )
            self.exercises[exercise.id] = exercise

    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        return self.exercises.get(exercise_id)

    def list_exercises(self) -> List[str]:
        return list(self.exercises.keys())

    def match_phase(self, exercise_id: str, current_angles: Dict[str, float]) -> Optional[Tuple[Phase, Dict[str, str]]]:
        exercise = self.get_exercise(exercise_id)
        if not exercise:
            return None

        best_phase = None
        best_violations = None
        best_violation_count = float('inf')

        for phase in exercise.phases:
            violations = {}
            for joint, angle_range in phase.angles.items():
                if joint in current_angles:
                    if not angle_range.contains(current_angles[joint]):
                        if current_angles[joint] < angle_range.min:
                            violations[joint] = f"too low ({current_angles[joint]:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"
                        else:
                            violations[joint] = f"too high ({current_angles[joint]:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"

            if len(violations) < best_violation_count:
                best_violation_count = len(violations)
                best_phase = phase
                best_violations = violations

        if best_phase:
            return best_phase, best_violations
        return None
