from typing import Dict, List, Optional
from core.exercise.exercise_model import ExerciseModel, Exercise, Phase, AngleRange


class ExerciseStateMachine:
    FRAMES_TO_TRANSITION = 5
    # TODO 4: Frames before triggering auto-recovery (approx 3 seconds at 30fps)
    FRAMES_TO_DISCONNECT = 90 

    # Note: We pass ExerciseModel so the state machine can use match_phase for recovery
    def __init__(self, exercise: Exercise, exercise_model: ExerciseModel):
        self.exercise = exercise
        self.exercise_model = exercise_model
        self.current_phase_index = 0
        self.rep_count = 0
        self.transition_counter = 0
        self.started = False
        self.missing_frames_counter = 0

    @property
    def current_phase(self) -> Phase:
        return self.exercise.phases[self.current_phase_index]

    @property
    def next_phase(self) -> Phase:
        next_index = (self.current_phase_index + 1) % len(self.exercise.phases)
        return self.exercise.phases[next_index]

    def update(self, angles: Dict[str, float]) -> dict:
        # TODO 4: Track total loss of tracking for Auto-Recovery
        if not angles:
            self.missing_frames_counter += 1
            return self._result(violations={}, transitioned=False)

        # Trigger auto-recovery if user was lost for a while and returns
        if self.missing_frames_counter >= self.FRAMES_TO_DISCONNECT and self.started:
            match = self.exercise_model.match_phase(self.exercise.id, angles)
            if match:
                matched_phase, _ = match
                try:
                    self.current_phase_index = self.exercise.phases.index(matched_phase)
                except ValueError:
                    pass
            self.transition_counter = 0

        # Reset disconnect counter since we have valid angles now
        self.missing_frames_counter = 0

        # Pre-workout: Check readiness to start
        if not self.started:
            readiness = self._check_readiness(angles, self.current_phase)
            
            # TODO 2: Ensure no mandatory joints are missing before starting
            if "missing" not in readiness.values() and all(status is None for status in readiness.values()):
                if self._angles_match_phase(angles, self.current_phase):
                    self.started = True
                    
            return self._result(violations={}, transitioned=False, readiness=readiness)

        # Active workout: Analyze merged rules (Global + Phase)
        active_rules = self._get_active_rules(self.current_phase)
        violations = self._check_violations(angles, active_rules)

        # Transition Logic (Strictly checks the next phase's requirements)
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
        """Strict check for transitions. Fails immediately if a required joint is missing."""
        for joint, angle_range in phase.angles.items():
            # TODO 1: Fix Vacuous Truth - Return False if joint is totally missing
            if joint not in angles:
                return False
            if not angle_range.contains(angles[joint]):
                return False
        return True

    def _get_active_rules(self, phase: Phase) -> Dict[str, AngleRange]:
        """TODO 3: Merges global safety constraints with the current phase's specific rules."""
        active_rules = self.exercise.global_constraints.copy()
        active_rules.update(phase.angles)
        return active_rules

    def _check_violations(self, angles: Dict[str, float], active_rules: Dict[str, AngleRange]) -> Dict[str, str]:
        """Checks actual angles against active rules (Global + Phase)"""
        violations = {}
        for joint, angle_range in active_rules.items():
            # If joint is missing during active phase, flag it
            if joint not in angles:
                violations[joint] = "missing (required)"
                continue
                
            if not angle_range.contains(angles[joint]):
                value = angles[joint]
                if value < angle_range.min:
                    violations[joint] = f"too low ({value:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"
                else:
                    violations[joint] = f"too high ({value:.0f}, expected {angle_range.min:.0f}-{angle_range.max:.0f})"
        return violations

    def _check_readiness(self, angles: Dict[str, float], phase: Phase) -> Dict[str, str]:
        """Validates starting position and presence of mandatory joints."""
        status = {}
        
        # TODO 2: Check if mandatory start joints are present
        for joint in self.exercise.mandatory_start_joints:
            if joint not in angles:
                status[joint] = "missing"
                
        # If mandatory joints are missing, return early to prompt user
        if "missing" in status.values():
            return status

        # Evaluate the angles for the starting phase
        for joint, angle_range in phase.angles.items():
            if joint in angles:
                val = angles[joint]
                status[joint] = None if angle_range.contains(val) else (
                    "too low" if val < angle_range.min else "too high"
                )
        return status

    def _result(self, violations: Dict[str, str], transitioned: bool,
                completed_rep: bool = False, readiness: Dict[str, str] = None) -> dict:
        return {
            'exercise': self.exercise.name,
            'phase': self.current_phase.name,
            'phase_index': self.current_phase_index,
            'phase_count': len(self.exercise.phases),
            'instruction': self.current_phase.instruction,
            'rep_count': self.rep_count,
            'violations': violations,
            'transitioned': transitioned,
            'completed_rep': completed_rep,
            'started': self.started,
            'readiness': readiness or {},
        }

    def reset(self):
        self.current_phase_index = 0
        self.rep_count = 0
        self.transition_counter = 0
        self.started = False
        self.missing_frames_counter = 0