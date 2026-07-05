import json
import os
from dataclasses import dataclass, field
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
    instruction: str = ""


@dataclass
class Exercise:
    id: str
    name: str
    description: str
    muscle_groups: List[str]
    phases: List[Phase]
    corrections: Dict[str, dict]
    
    # TODO 2: Mandatory Start Joints
    # Joints that MUST be visible by the camera to start the exercise.
    mandatory_start_joints: List[str] = field(default_factory=list)
    
    # TODO 3: Global Constraints
    # Safety rules that apply continuously across all phases (e.g., straight back).
    global_constraints: Dict[str, AngleRange] = field(default_factory=dict)

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
            # Note: adjust path logic if project structure changes
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'exercises.json')
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
                phases.append(Phase(
                    name=phase_data['name'],
                    order=phase_data['order'],
                    angles=angles,
                    instruction=phase_data.get('instruction', '')
                ))

            # TODO 3: Load global constraints from JSON
            global_constraints = {}
            if 'global_constraints' in ex_data:
                global_constraints = {
                    joint: AngleRange(min=r['min'], max=r['max'])
                    for joint, r in ex_data['global_constraints'].items()
                }

            exercise = Exercise(
                id=ex_data['id'],
                name=ex_data['name'],
                description=ex_data['description'],
                muscle_groups=ex_data['muscle_groups'],
                phases=phases,
                corrections=ex_data.get('corrections', {}),
                mandatory_start_joints=ex_data.get('mandatory_start_joints', []), # TODO 2
                global_constraints=global_constraints # TODO 3
            )
            self.exercises[exercise.id] = exercise

    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        return self.exercises.get(exercise_id)

    def list_exercises(self) -> List[str]:
        return list(self.exercises.keys())

    def match_phase(self, exercise_id: str, current_angles: Dict[str, float]) -> Optional[Tuple[Phase, Dict[str, str]]]:
        """
        Used for Auto-Recovery to find the closest matching phase index.
        Evaluates missing joints as errors to ensure accurate state guessing.
        """
        exercise = self.get_exercise(exercise_id)
        if not exercise:
            return None

        best_phase = None
        best_violations = None
        best_violation_count = float('inf')

        for phase in exercise.phases:
            violations = {}
            
            for joint, angle_range in phase.angles.items():
                # TODO 1/4: Treat missing required joints as violations for accurate phase matching
                if joint not in current_angles:
                    violations[joint] = "missing (required by phase)"
                elif not angle_range.contains(current_angles[joint]):
                    val = current_angles[joint]
                    direction = "too low" if val < angle_range.min else "too high"
                    violations[joint] = f"{direction} ({val:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"

            if len(violations) < best_violation_count:
                best_violation_count = len(violations)
                best_phase = phase
                best_violations = violations

        if best_phase:
            return best_phase, best_violations
        return None