// Runs the trained per-exercise RandomForest phase classifier (exported to
// ONNX by core/ml/export_onnx.py) client-side via onnxruntime-web.
//
// Originally wired up as a parallel signal only (shown in the HUD next to
// the rule-based phase, not trusted to drive anything) — see CLAUDE.md
// roadmap #10. That changed 2026-08-26: live A/B testing showed the ML
// counter far more accurate than the rule engine at rep counting, so
// mlRepCounter.ts now owns rep counting outright (see the comment in
// WorkoutScreen.tsx's live loop). This module itself is unchanged by that —
// it just classifies a phase per call — but it's no longer a side-channel:
// its output is load-bearing. Exercise picks its own model lazily on first use.
//
// Third stage of the pipeline: takes angles.ts's angles + deltaTracker.ts's
// velocities, runs them through the actual trained RandomForest (as ONNX,
// via onnxruntime-web — literally the same model file core/ml/trainer.py
// produced, not a re-implementation), and hands the raw per-call phase guess
// to mlRepCounter.ts, which smooths a sequence of these into stable
// transitions and an actual rep count.

// The '/wasm' subpath entry (CPU-only, no WebGPU/WebGL) — importing the bare
// 'onnxruntime-web' package pulls its WebGPU (jsep) backend's wasm variant
// into the production bundle via Rollup's static `new URL(...)` asset
// scanning, +27MB we never use. This entry point doesn't reference it.
import * as ort from 'onnxruntime-web/wasm';

// Served from public/ort/ (copied from node_modules/onnxruntime-web/dist/ —
// see frontend/README or the export_onnx workflow notes in CLAUDE.md).
// Only the base (non-jsep/jspi/asyncify) wasm+mjs pair is shipped — the
// others are WebGPU/threading variants this small model doesn't need and
// each one alone is 14-28MB.
ort.env.wasm.wasmPaths = '/ort/';

// Must match core/ml/trainer.py's JOINTS exactly — order defines feature
// position. right_wrist/left_wrist are listed there but never actually
// computed (core/ml/angles.py has no wrist angle definition either), so
// they're always the same missing-value sentinel on both sides — keep them
// here for alignment, not because they carry information.
const JOINTS = [
  'right_knee', 'left_knee',
  'right_hip', 'left_hip',
  'right_ankle', 'left_ankle',
  'right_elbow', 'left_elbow',
  'right_shoulder', 'left_shoulder',
  'right_wrist', 'left_wrist',
] as const;

export interface PhasePrediction {
  phase: string;
  confidence: number | null;   // null if the probability output couldn't be read
}

/** Same layout as core.ml.trainer.angles_to_features(): 12 angles then 12
 * per-joint deltas, in JOINTS order. Missing angle -> -1, missing delta -> 0. */
export function buildFeatures(
  angles: Record<string, number>,
  deltas: Record<string, number>,
): Float32Array {
  const out = new Float32Array(24);
  JOINTS.forEach((j, i) => { out[i] = angles[j] ?? -1; });
  JOINTS.forEach((j, i) => { out[12 + i] = deltas[j] ?? 0; });
  return out;
}

const sessions = new Map<string, Promise<ort.InferenceSession>>();

function loadSession(exerciseId: string): Promise<ort.InferenceSession> {
  let p = sessions.get(exerciseId);
  if (!p) {
    p = ort.InferenceSession.create(`/models/phase_${exerciseId}.onnx`, {
      executionProviders: ['wasm'],
    });
    sessions.set(exerciseId, p);
  }
  return p;
}

/** Preload a model so the first live prediction isn't the one paying for
 * the network fetch — call when the user picks an exercise, not per-frame. */
export function preloadPhaseModel(exerciseId: string): void {
  loadSession(exerciseId).catch(err => {
    console.error(`[phaseClassifier] failed to load model for ${exerciseId}:`, err);
    sessions.delete(exerciseId);
  });
}

export async function classifyPhase(
  exerciseId: string,
  angles: Record<string, number>,
  deltas: Record<string, number>,
): Promise<PhasePrediction | null> {
  let session: ort.InferenceSession;
  try {
    session = await loadSession(exerciseId);
  } catch (err) {
    console.error(`[phaseClassifier] session unavailable for ${exerciseId}:`, err);
    sessions.delete(exerciseId);
    return null;
  }

  const features = buildFeatures(angles, deltas);
  const tensor = new ort.Tensor('float32', features, [1, 24]);

  // export_onnx.py exports with skl2onnx's `zipmap=False` — 'probabilities'
  // is a plain [1, n_classes] float tensor instead of the ZipMap
  // (sequence<map<string,float>>) skl2onnx uses by default, which
  // onnxruntime-web's WASM backend can't marshal back to JS at all
  // (confirmed 1.29.0: session.run() throws for the whole call if a ZipMap
  // output is in the fetch list). Confidence = max probability, regardless
  // of which column is which class — order-independent, so no need to also
  // ship each exercise's class ordering to the frontend just for this.
  const outputs = await session.run({ features: tensor }, ['label', 'probabilities']);
  const phase = String(outputs.label.data[0]);
  const probsData = outputs.probabilities.data as Float32Array;
  const confidence = probsData.length ? Math.max(...probsData) : null;

  return { phase, confidence };
}
