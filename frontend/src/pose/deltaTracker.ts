// Per-joint rolling angular velocities for a live frame stream.
// Faithful port of core/ml/classifier.py's DeltaTracker — produces the same
// numbers (degrees/second) as the offline trainer.rolling_deltas(), so live
// inference sees the same feature distribution the model trained on,
// regardless of the browser's actual frame rate.

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
