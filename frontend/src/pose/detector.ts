// PoseLandmarker (MediaPipe Tasks JS) wrapper — loads the WASM runtime and
// the lite pose model once, then detects landmarks per video frame.
// This is the browser counterpart of stale/core/detection/pose_detector.py.
//
// First stage of the whole live pipeline: WorkoutScreen.tsx's frame loop
// calls `loadPoseLandmarker()` once (a module-level singleton — `landmarker`/
// `loading` cache the result so every exercise reuses the same downloaded
// model instead of re-fetching it) and then `detectPose()` on every frame.
// Its output (`DetectionResult`) is the raw material everything downstream
// consumes: `named` (landmarks keyed by name, using landmarks.ts's `LM`
// table) feeds angles.ts's `computeAngles()`, while `raw` (all 33 points by
// index, unfiltered) feeds the canvas skeleton drawing in WorkoutScreen.tsx.
// A `null` return means no person was detected this frame — every caller
// downstream has to handle that (an empty angles object, effectively).

import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { LM } from './landmarks';
import type { Landmarks, LandmarkName } from './landmarks';

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

let landmarker: PoseLandmarker | null = null;
let loading: Promise<PoseLandmarker> | null = null;

export function loadPoseLandmarker(): Promise<PoseLandmarker> {
  if (landmarker) return Promise.resolve(landmarker);
  if (loading) return loading;
  loading = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    return landmarker;
  })();
  return loading;
}

export interface DetectionResult {
  /** Named landmarks in normalized [0,1] coordinates. */
  named: Landmarks;
  /** All 33 raw landmarks (normalized), for skeleton drawing by index. */
  raw: { x: number; y: number; z: number; visibility: number }[];
}

export function detectPose(
  lm: PoseLandmarker, video: HTMLVideoElement, timestampMs: number,
): DetectionResult | null {
  const result = lm.detectForVideo(video, timestampMs);
  const pose = result.landmarks?.[0];
  if (!pose || pose.length === 0) return null;

  const named: Landmarks = {};
  for (const [name, idx] of Object.entries(LM)) {
    const p = pose[idx];
    if (!p) continue;
    named[name as LandmarkName] = {
      x: p.x, y: p.y, z: p.z ?? 0, visibility: p.visibility ?? 1,
    };
  }
  return {
    named,
    raw: pose.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0, visibility: p.visibility ?? 1 })),
  };
}
