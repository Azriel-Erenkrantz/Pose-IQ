// Per-joint rolling angular velocities for a live frame stream.
// Faithful port of core/ml/classifier.py's DeltaTracker — produces the same
// numbers (degrees/second) as the offline trainer.rolling_deltas(), so live
// inference sees the same feature distribution the model trained on,
// regardless of the browser's actual frame rate.
//
// Only exists for the ML path — the rule engine (stateMachine.ts) never
// looks at velocity at all, only the current angle value. The reason
// velocity matters here: a squat's "descending" and "ascending" phases can
// sit at the *exact same knee angle* mid-rep — the angle alone can't tell
// them apart, only which way the knee is currently moving can. That's why
// phaseClassifier.ts's feature vector is 24 numbers, not 12: the second half
// is this file's output. Wall-clock-normalized (uses real elapsed ms between
// calls, not a frame count) specifically so a browser tab that's dropped to
// 15fps under load still reports the same °/sec the model was trained to
// expect, rather than half the value.

const WINDOW = 5;

export class DeltaTracker {
  private prev: Record<string, number> | null = null;
  private prevT: number | null = null;
  private history: Record<string, number[]> = {};

  /** Feed the current frame's angles (degrees) and wall-clock ms; returns
   * {joint: mean velocity in degrees/second}. */
  update(angles: Record<string, number>, nowMs: number): Record<string, number> {
    if (this.prev !== null) {
      let dt = 1.0;
      if (this.prevT !== null) dt = (nowMs - this.prevT) / 1000;
      if (dt > 0) {
        for (const [joint, value] of Object.entries(angles)) {
          if (joint in this.prev) {
            const arr = this.history[joint] ?? (this.history[joint] = []);
            arr.push((value - this.prev[joint]) / dt);
            if (arr.length > WINDOW) arr.shift();
          }
        }
      }
    }
    this.prev = { ...angles };
    this.prevT = nowMs;

    const out: Record<string, number> = {};
    for (const [joint, arr] of Object.entries(this.history)) {
      if (arr.length > 0) out[joint] = arr.reduce((a, b) => a + b, 0) / arr.length;
    }
    return out;
  }
}
