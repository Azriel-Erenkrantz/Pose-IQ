// Deterministic phase state machine — faithful port of
// core/exercise/exercise_state_machine.py. The same Mongo-measured angle
// ranges drive both the desktop pipeline and this browser version.

import type { AngleRangeDef, ExerciseDef, PhaseDef } from '../api/types';

// ~5 frames (~165ms at the ~30fps live loop targets) was enough for a single
// noisy/jittery reading to falsely confirm a transition on real webcam
// tracking — confirmed live (2026-08-25): shoulder_press badly overcounted
// reps under real (imperfect lighting/framing) conditions, even after the
// same fix already applied to the started-readiness gate below. ~14 frames
// (~450ms+) is still well under a real rep's duration (at least ~1s) but
// filters out far more of that jitter.
const FRAMES_TO_TRANSITION = 14;
const FRAMES_TO_DISCONNECT = 90;   // ~3s at 30fps before auto-recovery kicks in

// Safety net, not a real fix: with only 2-3 front-view training clips per
// exercise, adjacent phases' measured angle ranges genuinely overlap a lot
// (confirmed live 2026-08-25 — shoulder_press's start/pressing/lowering
// elbow ranges overlap almost entirely) — a single continuous real
// repetition's motion can satisfy several phases' criteria in sequence
// within it, over-counting reps even with zero tracking noise at all. The
// real fix is more/richer training data narrowing those ranges. Until then,
// just refuse to count a rep faster than a human plausibly reps.
const MIN_FRAMES_BETWEEN_REPS = 30;   // ~1s at the ~30fps live loop target

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

// Tried three variants of "require more than just technically-in-range" to
// confirm a phase transition (all live-tested and reverted 2026-08-26, see
// [[static vs motion phase requirements]] memory for the full account):
// requiring exclusive entry, a symmetric core-range trim, then a direction-
// aware entry-edge-only trim. Each fixed the narrow case it targeted (a
// static mid-pause in an overlap zone free-wheeling into a fake rep) but
// made the far more common case — real full-range-of-motion reps — flaky or
// completely stuck, because the underlying measured ranges are just too
// noisy/overlapping (2-3 training clips per exercise) for any stricter
// geometric threshold to reliably help rather than hurt. Reverted to plain
// full-range matching below. The actual fix in progress is
// mlRepCounter.ts — an ML-classifier-driven rep counter running in
// parallel, to see whether it's more robust than any variant of this
// threshold-based approach can be.

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
  private startReadyCounter = 0;
  private framesSinceLastRep = MIN_FRAMES_BETWEEN_REPS;   // don't gate the very first rep
  private _dbgWasOk = false;   // TEMP (2026-08-26) — see console.log calls below, remove after diagnosis

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
    this.framesSinceLastRep += 1;

    // Pre-workout: wait until the user is in the starting position — held for
    // several CONSECUTIVE frames, same stability requirement as phase
    // transitions below. A single noisy frame (e.g. while still repositioning
    // the camera) used to be enough to flip `started`, after which further
    // incidental movement could rack up spurious reps; requiring a sustained
    // match closes that gap the same way transitionCounter already does.
    if (!this.started) {
      const readiness = this.checkReadiness(angles, this.currentPhase);
      const values = Object.values(readiness);
      const ok = !values.includes('missing') && values.every(s => s === null)
        && this.anglesMatchPhase(angles, this.currentPhase);
      if (ok) {
        this.startReadyCounter += 1;
        if (this.startReadyCounter >= FRAMES_TO_TRANSITION) {
          this.started = true;
          // eslint-disable-next-line no-console
          console.log('[SM-DEBUG] started=true', 'angles=', angles);
        }
      } else {
        // TEMP diagnostic (2026-08-26) — only logs when readiness *stops*
        // being satisfied, so it's one line per real failure, not spam.
        if (this._dbgWasOk) {
          const bad = Object.entries(readiness).filter(([, s]) => s !== null);
          // eslint-disable-next-line no-console
          console.log('[SM-DEBUG] readiness lost, counter reset. failing:', bad, 'angles=', angles);
        }
        this.startReadyCounter = 0;
      }
      this._dbgWasOk = ok;
      return this.result(false, false, readiness);
    }

    // Transition when the NEXT phase's requirements hold for several
    // CONSECUTIVE frames. Originally this only cost 1 point of progress per
    // non-matching frame instead of resetting outright, on the theory that a
    // single dropped frame shouldn't wipe out an otherwise-good run — but
    // live testing (2026-08-25) showed that during *real* exercise motion,
    // natural tremor/imprecision near a phase boundary oscillates in and out
    // of range often enough to still slowly net-accumulate toward the
    // threshold without ever being truly stable, badly overcounting reps.
    // A hard reset demands genuinely consecutive frames — still under ~0.5s
    // of real stability at the current threshold, cheap even if an
    // occasional single dropped frame costs the whole run.
    //
    // Full range, plain — not the core-range/exclusive-entry variants tried
    // and reverted 2026-08-26 (see the comment above `contains`). Those
    // targeted a real but narrower bug (a static mid-pause in an overlap
    // zone free-wheeling into an extra rep) at the cost of getting
    // genuinely stuck on ordinary full-effort reps far more often — a worse
    // trade given how thin the training data already makes these ranges.
    if (this.anglesMatchPhase(angles, this.nextPhase)) {
      this.transitionCounter = Math.max(this.transitionCounter, 0) + 1;
      if (this.transitionCounter >= FRAMES_TO_TRANSITION) {
        return this.advance();
      }
    } else {
      // TEMP diagnostic (2026-08-26) — one line per real failure (progress
      // was building, then broke), showing exactly which joint(s) in the
      // *next* phase's required range the current angles miss and by how
      // much, instead of guessing why a transition won't confirm.
      if (this.transitionCounter > 0) {
        const misses = Object.entries(this.nextPhase.angles)
          .filter(([joint, range]) => !(joint in angles) || !contains(range, angles[joint]))
          .map(([joint, range]) => `${joint}: got ${angles[joint]?.toFixed(1) ?? 'missing'}, need [${range.min.toFixed(1)}, ${range.max.toFixed(1)}]`);
        // eslint-disable-next-line no-console
        console.log(`[SM-DEBUG] stuck on "${this.currentPhase.name}", next="${this.nextPhase.name}" failed:`, misses);
      }
      this.transitionCounter = 0;
    }

    return this.result(false);
  }

  private advance(): StateMachineResult {
    const fromPhase = this.currentPhase.name;
    this.transitionCounter = 0;
    this.currentPhaseIndex = (this.currentPhaseIndex + 1) % this.exercise.phases.length;
    const wrapped = this.currentPhaseIndex === 0;
    // eslint-disable-next-line no-console
    console.log(`[SM-DEBUG] advance ${fromPhase} -> ${this.currentPhase.name}`,
      `wrapped=${wrapped}`, `framesSinceLastRep=${this.framesSinceLastRep}`);
    // Phase still cycles normally either way (so the UI/instruction stays
    // live and the machine doesn't stall) — only the *count* is gated, so a
    // too-fast wrap silently doesn't score, rather than blocking anything.
    const completedRep = wrapped && this.framesSinceLastRep >= MIN_FRAMES_BETWEEN_REPS;
    if (completedRep) {
      this.repCount += 1;
      this.framesSinceLastRep = 0;
    }
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
    this.startReadyCounter = 0;
    this.framesSinceLastRep = MIN_FRAMES_BETWEEN_REPS;
  }
}
