// Deterministic phase state machine — faithful port of
// core/exercise/exercise_state_machine.py. The same Mongo-measured angle
// ranges drive both the desktop pipeline and this browser version.

import type { AngleRangeDef, ExerciseDef, PhaseDef } from '../api/types';

const FRAMES_TO_TRANSITION = 5;
const FRAMES_TO_DISCONNECT = 90;   // ~3s at 30fps before auto-recovery kicks in

export type ReadinessStatus = 'missing' | 'too low' | 'too high' | null;

export interface StateMachineResult {
  phase: string;
  phaseIndex: number;
  phaseCount: number;
  instruction: string;
  repCount: number;
  violations: Record<string, string>;
  transitioned: boolean;
  completedRep: boolean;
  started: boolean;
  readiness: Record<string, ReadinessStatus>;
}

function contains(r: AngleRangeDef, value: number): boolean {
  return r.min <= value && value <= r.max;
}

export class ExerciseStateMachine {
  private exercise: ExerciseDef;
  private currentPhaseIndex = 0;
  repCount = 0;
  private transitionCounter = 0;
  started = false;
  private missingFramesCounter = 0;

  constructor(exercise: ExerciseDef) {
    this.exercise = exercise;
  }

  get currentPhase(): PhaseDef {
    return this.exercise.phases[this.currentPhaseIndex];
  }

  private get nextPhase(): PhaseDef {
    return this.exercise.phases[(this.currentPhaseIndex + 1) % this.exercise.phases.length];
  }

  /** Global safety constraints merged with the current phase's rules. */
  activeRules(): Record<string, AngleRangeDef> {
    return { ...this.exercise.global_constraints, ...this.currentPhase.angles };
  }

  update(angles: Record<string, number>): StateMachineResult {
    if (Object.keys(angles).length === 0) {
      this.missingFramesCounter += 1;
      return this.result({}, false);
    }

    // Auto-recovery: user was lost for a while and is back — re-sync to the
    // best-matching phase instead of deadlocking on the stale one.
    if (this.missingFramesCounter >= FRAMES_TO_DISCONNECT && this.started) {
      const matched = this.matchPhase(angles);
      if (matched !== null) this.currentPhaseIndex = matched;
      this.transitionCounter = 0;
    }
    this.missingFramesCounter = 0;

    // Pre-workout: wait until the user is in the starting position.
    if (!this.started) {
      const readiness = this.checkReadiness(angles, this.currentPhase);
      const values = Object.values(readiness);
      if (!values.includes('missing') && values.every(s => s === null)) {
        if (this.anglesMatchPhase(angles, this.currentPhase)) {
          this.started = true;
        }
      }
      return this.result({}, false, false, readiness);
    }

    const violations = this.checkViolations(angles, this.activeRules());

    // Transition when the NEXT phase's requirements hold for several frames.
    if (this.anglesMatchPhase(angles, this.nextPhase)) {
      this.transitionCounter += 1;
      if (this.transitionCounter >= FRAMES_TO_TRANSITION) {
        return this.advance();
      }
    } else {
      this.transitionCounter = 0;
    }

    return this.result(violations, false);
  }

  private advance(): StateMachineResult {
    this.transitionCounter = 0;
    this.currentPhaseIndex = (this.currentPhaseIndex + 1) % this.exercise.phases.length;
    const completedRep = this.currentPhaseIndex === 0;
    if (completedRep) this.repCount += 1;
    return this.result({}, true, completedRep);
  }

  private anglesMatchPhase(angles: Record<string, number>, phase: PhaseDef): boolean {
    for (const [joint, range] of Object.entries(phase.angles)) {
      if (!(joint in angles)) return false;
      if (!contains(range, angles[joint])) return false;
    }
    return true;
  }

  private checkViolations(
    angles: Record<string, number>,
    rules: Record<string, AngleRangeDef>,
  ): Record<string, string> {
    const violations: Record<string, string> = {};
    for (const [joint, range] of Object.entries(rules)) {
      if (!(joint in angles)) {
        violations[joint] = 'missing (required)';
        continue;
      }
      const value = angles[joint];
      if (!contains(range, value)) {
        violations[joint] = value < range.min ? 'too low' : 'too high';
      }
    }
    return violations;
  }

  private checkReadiness(
    angles: Record<string, number>,
    phase: PhaseDef,
  ): Record<string, ReadinessStatus> {
    const status: Record<string, ReadinessStatus> = {};

    for (const joint of this.exercise.mandatory_start_joints) {
      if (!(joint in angles)) status[joint] = 'missing';
    }
    if (Object.values(status).includes('missing')) return status;

    for (const [joint, range] of Object.entries(phase.angles)) {
      if (joint in angles) {
        const val = angles[joint];
        status[joint] = contains(range, val) ? null : (val < range.min ? 'too low' : 'too high');
      }
    }
    return status;
  }

  /** Best-matching phase index by fewest violations (occlusion-robust). */
  private matchPhase(angles: Record<string, number>): number | null {
    let bestIndex: number | null = null;
    let bestScore = Infinity;

    this.exercise.phases.forEach((phase, i) => {
      const diagRanges: Record<string, AngleRangeDef> = {};
      for (const j of phase.diagnostic_joints) {
        if (j in phase.angles) diagRanges[j] = phase.angles[j];
      }
      const check = Object.keys(diagRanges).length ? diagRanges : phase.angles;

      let violations = 0;
      for (const [joint, range] of Object.entries(check)) {
        if (!(joint in angles) || !contains(range, angles[joint])) violations += 1;
      }
      if (violations < bestScore) {
        bestScore = violations;
        bestIndex = i;
      }
    });

    return bestIndex;
  }

  private result(
    violations: Record<string, string>,
    transitioned: boolean,
    completedRep = false,
    readiness: Record<string, ReadinessStatus> = {},
  ): StateMachineResult {
    return {
      phase: this.currentPhase.name,
      phaseIndex: this.currentPhaseIndex,
      phaseCount: this.exercise.phases.length,
      instruction: this.currentPhase.instruction,
      repCount: this.repCount,
      violations,
      transitioned,
      completedRep,
      started: this.started,
      readiness,
    };
  }

  reset(): void {
    this.currentPhaseIndex = 0;
    this.repCount = 0;
    this.transitionCounter = 0;
    this.started = false;
    this.missingFramesCounter = 0;
  }
}
