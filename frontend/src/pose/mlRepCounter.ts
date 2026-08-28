// Counts reps from the ONNX phase classifier's predictions alone,
// independent of the angle-range rule engine (see CLAUDE.md roadmap #10).
// Confirmed 2026-08-26 (repeated 10-rep sets) far more accurate than the
// rule engine, which kept getting stuck on overlapping boundary ranges —
// this is now the real rep-counting authority, not an experiment. Still
// gated by the rule engine's readiness check for *when* to start tracking.

import type { ExerciseDef } from '../api/types';

// classifyPhase runs throttled (~250ms, see WorkoutScreen's
// ML_DEBUG_INTERVAL_MS) — not a fixed-rate 30fps loop like the rule engine,
// so the min-time-between-reps safety net (mirroring stateMachine.ts's
// MIN_FRAMES_BETWEEN_REPS) is wall-clock ms instead of a frame count.
const MIN_MS_BETWEEN_REPS = 1000;

// First attempt (reverted same day) required N *consecutive* matching
// predictions, mirroring stateMachine.ts's hard-reset transition counter.
// That's a bad fit here: confirmed live 2026-08-26, the model reliably
// nails "start"->"pressing" and "pressing"->"lowering" (clean 2-in-a-row),
// but at "lowering"->"start" — the same bottom-of-rep boundary the rule
// engine's ranges are already known to overlap on — it genuinely flickers
// tick-to-tick between "start" and "pressing" for a few ticks before
// settling. At a slow ~4Hz tick rate, one hard reset there throws away up
// to ~500ms of real progress, so it never won a race against a fresh
// flicker and the rep never counted. A majority vote over a short window
// tolerates that flicker while still requiring real, sustained evidence —
// the standard smoothing technique for a noisy per-tick classifier, unlike
// stateMachine.ts's continuous 30fps signal where consecutive-frame
// stability made sense.
const WINDOW_SIZE = 5;
const MAJORITY_NEEDED = 3;   // ~3 of the last ~5 ticks (~1.25s window)

// A first guess, not yet tuned against real data — this experiment is partly
// to find out what threshold (if any) is actually needed. Below this, a
// prediction is treated as noise: doesn't advance, but also doesn't reset
// progress toward a transition that's already building.
const MIN_CONFIDENCE = 0.4;

// None of this exercise's trained classes represent "not exercising" — the
// classifier always has to pick one of start/pressing/lowering even for a
// frame where the user has simply stopped and is sitting still (confirmed
// live 2026-08-26: after finishing a real set, idling in frame produced an
// occasional stray rep every few seconds). A real rep requires sustained
// joint motion — this doesn't distinguish idle drift from a genuine slow
// rep, but it does filter the common case of near-zero motion, which the
// deltas already computed as model input features conveniently also serve
// to check here.
const MIN_MOTION_DEG_PER_SEC = 5;

export interface MlCountResult {
  phase: string;
  phaseIndex: number;
  repCount: number;
  completedRep: boolean;
}

export class MlRepCounter {
  // Excludes 'stable' (is_initial) phases like "start" from the tracked
  // cycle — mid-set, the model rarely predicts "start" again (see below),
  // and a rep is complete once you're pressing again after lowering anyway.
  private motionPhases: string[];
  private currentIndex = 0;
  repCount = 0;
  private history: string[] = [];
  private lastRepAt = 0;

  constructor(exercise: ExerciseDef) {
    this.motionPhases = exercise.phases
      .filter(p => p.motion_direction !== 'stable')
      .sort((a, b) => a.order - b.order)
      .map(p => p.name);
  }

  private get currentPhaseName(): string {
    return this.motionPhases[this.currentIndex];
  }

  private get nextPhaseName(): string {
    return this.motionPhases[(this.currentIndex + 1) % this.motionPhases.length];
  }

  update(predictedPhase: string, confidence: number | null, now: number, maxMotion: number): MlCountResult {
    if (confidence !== null && confidence < MIN_CONFIDENCE) {
      return this.result(false);   // too unsure — ignore this tick, don't reset progress either
    }
    if (maxMotion < MIN_MOTION_DEG_PER_SEC) {
      return this.result(false);   // basically motionless — filters idle drift, not a real rep
    }

    this.history.push(predictedPhase);
    if (this.history.length > WINDOW_SIZE) this.history.shift();

    // Majority vote: has the *next* phase in the cycle won most of the
    // recent window, not just this one tick?
    const nextVotes = this.history.filter(p => p === this.nextPhaseName).length;
    // eslint-disable-next-line no-console
    console.log(`[ML-DEBUG] "${this.currentPhaseName}" -> "${this.nextPhaseName}"? votes ${nextVotes}/${MAJORITY_NEEDED} (last tick: "${predictedPhase}")`);
    if (nextVotes >= MAJORITY_NEEDED) {
      return this.advance(now);
    }

    return this.result(false);
  }

  // Moves to the next phase in the cycle; only counts a rep if that move
  // wraps back to the start AND enough real time passed since the last one.
  private advance(now: number): MlCountResult {
    this.history = [];
    this.currentIndex = (this.currentIndex + 1) % this.motionPhases.length;
    const wrapped = this.currentIndex === 0;
    const completedRep = wrapped && (now - this.lastRepAt >= MIN_MS_BETWEEN_REPS);
    // eslint-disable-next-line no-console
    console.log(`[ML-DEBUG] advance -> "${this.currentPhaseName}" wrapped=${wrapped} completedRep=${completedRep}`);
    if (completedRep) {
      this.repCount += 1;
      this.lastRepAt = now;
    }
    return this.result(completedRep);
  }

  private result(completedRep: boolean): MlCountResult {
    return {
      phase: this.currentPhaseName,
      phaseIndex: this.currentIndex,
      repCount: this.repCount,
      completedRep,
    };
  }
}
