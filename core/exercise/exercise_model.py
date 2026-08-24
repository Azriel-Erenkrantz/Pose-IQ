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


# ---------------------------------------------------------------------------
# Shared assembly helpers — both loaders (JSON seed file and Mongo) build the
# same Phase/Exercise shape; only *where the angle numbers come from* differs
# (embedded in the seed JSON vs. a separate exercise_angles collection), so
# that's the only part each loader implements itself.
# ---------------------------------------------------------------------------

def _build_global_constraints(raw: Dict[str, dict]) -> Dict[str, AngleRange]:
    return {
        joint: AngleRange(min=r['min'], max=r['max'], corrections=r.get('corrections', {}))
        for joint, r in raw.items()
    }


def _derive_primary_joints(phases_data: List[dict]) -> List[str]:
    """Diagnostic joints of every non-initial phase, in first-seen order —
    these are the joints whose angles actually distinguish this exercise's
    movement phases (the initial/resting phase is excluded since it doesn't
    help tell one exercise's reps apart from another's)."""
    seen: set = set()
    primary: List[str] = []
    for phase_data in phases_data:
        if phase_data.get('is_initial', False):
            continue
        for joint in phase_data.get('diagnostic_joints', []):
            if joint not in seen:
                seen.add(joint)
                primary.append(joint)
    return primary


def _build_phase(phase_data: dict, angles: Dict[str, AngleRange]) -> Phase:
    return Phase(
        name=phase_data['name'],
        order=phase_data['order'],
        angles=angles,
        instruction=phase_data.get('instruction', ''),
        is_initial=phase_data.get('is_initial', False),
        diagnostic_joints=phase_data.get('diagnostic_joints', []),
        motion_direction=phase_data.get('motion_direction', 'stable'),
    )


def _assemble_exercise(ex_data: dict, phases: List[Phase]) -> Exercise:
    return Exercise(
        id=ex_data['id'],
        name=ex_data['name'],
        description=ex_data['description'],
        muscle_groups=ex_data['muscle_groups'],
        phases=phases,
        primary_joints=_derive_primary_joints(ex_data.get('phases', [])),
        mandatory_start_joints=ex_data.get('mandatory_start_joints', []),
        global_constraints=_build_global_constraints(ex_data.get('global_constraints', {})),
    )


def _seed_phase_angles(phase_data: dict) -> Dict[str, AngleRange]:
    """Angle ranges straight from the seed JSON — supports both the bare
    seed format (joints: {corrections only}, no numbers yet) and the fuller
    format (angle_ranges: {min, max, mean, std, corrections})."""
    raw = phase_data.get('angle_ranges', phase_data.get('joints', {}))
    angles = {}
    for joint, r in raw.items():
        if 'min' not in r:
            continue  # seed format — no angle data yet, skip
        angles[joint] = AngleRange(
            min=r['min'], max=r['max'], corrections=r.get('corrections', {}),
            mean=r.get('mean'), std=r.get('std'),
        )
    return angles


def _mongo_phase_angles(phase_data: dict, measured: Dict[str, dict]) -> Dict[str, AngleRange]:
    """Angle ranges for one phase, joining the exercise's phase/joint
    structure (phase_data) against measured min/max/mean/std for that phase
    (measured, from the exercise_angles collection) — a joint with no
    measurement yet (trainer.py hasn't run) is skipped, same as the seed
    loader skips joints with no numbers."""
    angles = {}
    for joint, joint_def in phase_data.get('joints', {}).items():
        stats = measured.get(joint)
        if stats is None:
            continue
        angles[joint] = AngleRange(
            min=stats['min'], max=stats['max'], corrections=joint_def.get('corrections', {}),
            mean=stats.get('mean'), std=stats.get('std'),
        )
    return angles


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
            phases = [
                _build_phase(phase_data, _seed_phase_angles(phase_data))
                for phase_data in sorted(ex_data['phases'], key=lambda p: p['order'])
            ]
            exercise = _assemble_exercise(ex_data, phases)
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
            # Measured angle stats for this exercise, keyed by phase name —
            # looked up per-phase below when building each Phase's angles.
            angle_by_phase: Dict[str, dict] = {
                doc['phase']: doc['joints']
                for doc in db.exercise_angles.find({'exercise_id': ex_doc['id']})
            }

            phases = [
                _build_phase(phase_data, _mongo_phase_angles(phase_data, angle_by_phase.get(phase_data['name'], {})))
                for phase_data in sorted(ex_doc.get('phases', []), key=lambda p: p['order'])
            ]
            exercise = _assemble_exercise(ex_doc, phases)
            instance.exercises[exercise.id] = exercise

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
