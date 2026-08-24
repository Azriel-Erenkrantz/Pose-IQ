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
  transitioned: boolean;
  completedRep: boolean;
  started: boolean;
  readiness: Record<string, ReadinessStatus>;
}

export function contains(r: AngleRangeDef, value: number): boolean {
  return r.min <= value && value <= r.max;
}

/** null when in range, otherwise which way it's out of bounds. */
function direction(r: AngleRangeDef, value: number): 'too low' | 'too high' | null {
  if (contains(r, value)) return null;
  return value < r.min ? 'too low' : 'too high';
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
      return this.result(false);
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
      return this.result(false, false, readiness);
    }

    // Transition when the NEXT phase's requirements hold for several NET
    // frames — a single noisy frame only costs 1 point of progress rather
    // than wiping out an otherwise-good run, so brief tracking jitter mid-rep
    // doesn't force starting the count over from zero.
    if (this.anglesMatchPhase(angles, this.nextPhase)) {
      this.transitionCounter += 1;
      if (this.transitionCounter >= FRAMES_TO_TRANSITION) {
        return this.advance();
      }
    } else {
      this.transitionCounter = Math.max(0, this.transitionCounter - 1);
    }

    return this.result(false);
  }

  private advance(): StateMachineResult {
    this.transitionCounter = 0;
    this.currentPhaseIndex = (this.currentPhaseIndex + 1) % this.exercise.phases.length;
    const completedRep = this.currentPhaseIndex === 0;
    if (completedRep) this.repCount += 1;
    return this.result(true, completedRep);
  }

  private anglesMatchPhase(angles: Record<string, number>, phase: PhaseDef): boolean {
    for (const [joint, range] of Object.entries(phase.angles)) {
      if (!(joint in angles)) return false;
      if (!contains(range, angles[joint])) return false;
    }
    return true;
  }

  private checkReadiness(
    angles: Record<string, number>,
    phase: PhaseDef,
  ): Record<string, ReadinessStatus> {
    const status: Record<string, ReadinessStatus> = {};

    // Joints the phase itself needs are just as required as the exercise's
    // declared mandatory ones — flag both, so a missing joint always shows
    // up here instead of silently blocking anglesMatchPhase() with no
    // explanation the user (or the UI) can see.
    const required = new Set([...this.exercise.mandatory_start_joints, ...Object.keys(phase.angles)]);
    for (const joint of required) {
      if (!(joint in angles)) {
        status[joint] = 'missing';
      } else if (joint in phase.angles) {
        status[joint] = direction(phase.angles[joint], angles[joint]);
      } else {
        status[joint] = null;
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
