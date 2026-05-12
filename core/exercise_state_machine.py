from typing import Dict, List, Optional
from exercise_model import ExerciseModel, Exercise, Phase


class ExerciseStateMachine:
    FRAMES_TO_TRANSITION = 5

    def __init__(self, exercise: Exercise):
        self.exercise = exercise
        self.current_phase_index = 0
        self.rep_count = 0
        self.transition_counter = 0
        self.started = False

    @property
    def current_phase(self) -> Phase:
        return self.exercise.phases[self.current_phase_index]

    @property
    def next_phase(self) -> Phase:
        next_index = (self.current_phase_index + 1) % len(self.exercise.phases)
        return self.exercise.phases[next_index]

    def update(self, angles: Dict[str, float]) -> dict:
        if not angles:
            return self._result(violations={}, transitioned=False)

        if not self.started:
            if self._angles_match_phase(angles, self.current_phase):
                self.started = True
            return self._result(violations={}, transitioned=False)

        violations = self._check_violations(angles, self.current_phase)

        if self._angles_match_phase(angles, self.next_phase):
            self.transition_counter += 1
            if self.transition_counter >= self.FRAMES_TO_TRANSITION:
                return self._advance()
        else:
            self.transition_counter = 0

        return self._result(violations=violations, transitioned=False)

    def _advance(self) -> dict:
        self.transition_counter = 0
        self.current_phase_index = (self.current_phase_index + 1) % len(self.exercise.phases)

        completed_rep = False
        if self.current_phase_index == 0:
            self.rep_count += 1
            completed_rep = True

        return self._result(violations={}, transitioned=True, completed_rep=completed_rep)

    def _angles_match_phase(self, angles: Dict[str, float], phase: Phase) -> bool:
        for joint, angle_range in phase.angles.items():
            if joint in angles and not angle_range.contains(angles[joint]):
                return False
        return True

    def _check_violations(self, angles: Dict[str, float], phase: Phase) -> Dict[str, str]:
        violations = {}
        for joint, angle_range in phase.angles.items():
            if joint in angles and not angle_range.contains(angles[joint]):
                value = angles[joint]
                if value < angle_range.min:
                    violations[joint] = f"too low ({value:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"
                else:
                    violations[joint] = f"too high ({value:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"
        return violations

    def _result(self, violations: Dict[str, str], transitioned: bool, completed_rep: bool = False) -> dict:
        return {
            'exercise': self.exercise.name,
            'phase': self.current_phase.name,
            'rep_count': self.rep_count,
            'violations': violations,
            'transitioned': transitioned,
            'completed_rep': completed_rep,
            'started': self.started
        }

    def reset(self):
        self.current_phase_index = 0
        self.rep_count = 0
        self.transition_counter = 0
        self.started = False
