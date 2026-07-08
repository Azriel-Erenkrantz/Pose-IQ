import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AngleRange:
    min: float
    max: float
    corrections: Dict[str, str] = field(default_factory=dict)  # too_low, too_high, severity
    mean: Optional[float] = None   # measured from reference videos
    std: Optional[float] = None    # measured from reference videos

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max

    def correction_message(self, value: float) -> Tuple[str, str]:
        """Return (message, severity) for a value outside this range."""
        direction = 'too_low' if value < self.min else 'too_high'
        severity = self.corrections.get('severity', 'medium')
        message = self.corrections.get(
            direction,
            f"{direction.replace('_', ' ')} ({value:.0f}°, expected {self.min:.0f}–{self.max:.0f}°)"
        )
        return message, severity


@dataclass
class Phase:
    name: str
    order: int
    angles: Dict[str, AngleRange]
    instruction: str = ""
    is_initial: bool = False
    diagnostic_joints: List[str] = field(default_factory=list)
    motion_direction: str = "stable"   # 'stable' | 'increasing' | 'decreasing'


@dataclass
class Exercise:
    id: str
    name: str
    description: str
    muscle_groups: List[str]
    phases: List[Phase]
    primary_joints: List[str] = field(default_factory=list)
    mandatory_start_joints: List[str] = field(default_factory=list)
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

    def initial_phase(self) -> Optional[Phase]:
        for phase in self.phases:
            if phase.is_initial:
                return phase
        return self.phases[0] if self.phases else None


class ExerciseModel:
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'exercises_seed.json'
            )
        self.exercises: Dict[str, Exercise] = {}
        self._load(data_path)

    def _load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for ex_data in data['exercises']:
            phases = []
            for phase_data in sorted(ex_data['phases'], key=lambda p: p['order']):
                # Support both seed format (joints: {corrections only}) and
                # full format (angle_ranges: {min, max, mean, std, corrections})
                angles = {}
                for joint, r in phase_data.get('angle_ranges', phase_data.get('joints', {})).items():
                    if 'min' not in r:
                        continue  # seed format — no angle data yet, skip
                    angles[joint] = AngleRange(
                        min=r['min'],
                        max=r['max'],
                        corrections=r.get('corrections', {}),
                        mean=r.get('mean'),
                        std=r.get('std'),
                    )
                phases.append(Phase(
                    name=phase_data['name'],
                    order=phase_data['order'],
                    angles=angles,
                    instruction=phase_data.get('instruction', ''),
                    is_initial=phase_data.get('is_initial', False),
                    diagnostic_joints=phase_data.get('diagnostic_joints', []),
                    motion_direction=phase_data.get('motion_direction', 'stable'),
                ))

            global_constraints = {
                joint: AngleRange(
                    min=r['min'],
                    max=r['max'],
                    corrections=r.get('corrections', {}),
                )
                for joint, r in ex_data.get('global_constraints', {}).items()
            }

            # Derive primary_joints from diagnostic joints of non-initial phases
            seen: set = set()
            primary_joints: List[str] = []
            for ph in ex_data.get('phases', []):
                if not ph.get('is_initial', False):
                    for j in ph.get('diagnostic_joints', []):
                        if j not in seen:
                            seen.add(j)
                            primary_joints.append(j)

            exercise = Exercise(
                id=ex_data['id'],
                name=ex_data['name'],
                description=ex_data['description'],
                muscle_groups=ex_data['muscle_groups'],
                phases=phases,
                primary_joints=primary_joints,
                mandatory_start_joints=ex_data.get('mandatory_start_joints', []),
                global_constraints=global_constraints,
            )
            self.exercises[exercise.id] = exercise

    @classmethod
    def from_mongo(cls, db) -> 'ExerciseModel':
        """
        Load exercises from MongoDB.

        Merges two collections:
          exercises       — metadata, corrections, phase structure (no angles)
          exercise_angles — min/max/mean/std per phase+joint from Model 2
        """
        instance = cls.__new__(cls)
        instance.exercises = {}

        for ex_doc in db.exercises.find():
            ex_id = ex_doc['id']

            # Load all angle data for this exercise keyed by phase name
            angle_by_phase: Dict[str, dict] = {
                doc['phase']: doc['joints']
                for doc in db.exercise_angles.find({'exercise_id': ex_id})
            }

            phases = []
            for phase_data in sorted(ex_doc.get('phases', []), key=lambda p: p['order']):
                phase_angle_data = angle_by_phase.get(phase_data['name'], {})
                angles: Dict[str, AngleRange] = {}

                for joint, joint_def in phase_data.get('joints', {}).items():
                    stats = phase_angle_data.get(joint)
                    if stats is None:
                        # No measured data yet — skip until Model 2 has run
                        continue
                    angles[joint] = AngleRange(
                        min=stats['min'],
                        max=stats['max'],
                        corrections=joint_def.get('corrections', {}),
                        mean=stats.get('mean'),
                        std=stats.get('std'),
                    )

                phases.append(Phase(
                    name=phase_data['name'],
                    order=phase_data['order'],
                    angles=angles,
                    instruction=phase_data.get('instruction', ''),
                    is_initial=phase_data.get('is_initial', False),
                    diagnostic_joints=phase_data.get('diagnostic_joints', []),
                    motion_direction=phase_data.get('motion_direction', 'stable'),
                ))

            # Global constraints keep their min/max in the exercises collection
            global_constraints: Dict[str, AngleRange] = {
                joint: AngleRange(
                    min=r['min'],
                    max=r['max'],
                    corrections=r.get('corrections', {}),
                )
                for joint, r in ex_doc.get('global_constraints', {}).items()
            }

            # primary_joints = diagnostic joints across all non-initial phases (in order)
            seen: set = set()
            pj: List[str] = []
            for ph_data in ex_doc.get('phases', []):
                if not ph_data.get('is_initial', False):
                    for j in ph_data.get('diagnostic_joints', []):
                        if j not in seen:
                            seen.add(j)
                            pj.append(j)

            exercise = Exercise(
                id=ex_id,
                name=ex_doc['name'],
                description=ex_doc['description'],
                muscle_groups=ex_doc['muscle_groups'],
                phases=phases,
                primary_joints=pj,
                mandatory_start_joints=ex_doc.get('mandatory_start_joints', []),
                global_constraints=global_constraints,
            )
            instance.exercises[ex_id] = exercise

        return instance

    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        return self.exercises.get(exercise_id)

    def list_exercises(self) -> List[str]:
        return list(self.exercises.keys())

    def match_phase(self, exercise_id: str, current_angles: Dict[str, float]) -> Optional[Tuple[Phase, Dict[str, str]]]:
        """
        Find the best-matching phase by fewest joint violations.
        Uses diagnostic_joints when available for occlusion robustness.
        """
        exercise = self.get_exercise(exercise_id)
        if not exercise:
            return None

        best_phase = None
        best_violations: Dict[str, str] = {}
        best_score = float('inf')

        for phase in exercise.phases:
            check_joints = (
                {j: phase.angles[j] for j in phase.diagnostic_joints if j in phase.angles}
                or phase.angles
            )
            violations = {}
            for joint, angle_range in check_joints.items():
                if joint not in current_angles:
                    violations[joint] = 'missing'
                elif not angle_range.contains(current_angles[joint]):
                    val = current_angles[joint]
                    violations[joint] = 'too_low' if val < angle_range.min else 'too_high'

            if len(violations) < best_score:
                best_score = len(violations)
                best_phase = phase
                best_violations = violations

        return (best_phase, best_violations) if best_phase else None
