// PoseLandmarker (MediaPipe Tasks JS) wrapper — loads the WASM runtime and
// the lite pose model once, then detects landmarks per video frame.
// This is the browser counterpart of stale/core/detection/pose_detector.py.

import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { LM } from './landmarks';
import type { Landmarks, LandmarkName } from './landmarks';

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// Module-level singleton — `landmarker`/`loading` cache the loaded model so
// switching exercises reuses it instead of re-downloading.
let landmarker: PoseLandmarker | null = null;
let loading: Promise<PoseLandmarker> | null = null;

// Call once before the live loop starts; safe to call again later, returns
// the cached instance/in-flight promise instead of loading twice.
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

// Runs MediaPipe on one video frame. Returns null when no person is
// detected — callers treat that as "no angles this frame", not an error.
export function detectPose(
  lm: PoseLandmarker, video: HTMLVideoElement, timestampMs: number,
): DetectionResult | null {
  const result = lm.detectForVideo(video, timestampMs);
  const pose = result.landmarks?.[0];   // numPoses: 1, so only index 0 is ever populated
  if (!pose || pose.length === 0) return null;

  // Named view: only the joints this app actually tracks (via LM), keyed by
  // name — what angles.ts consumes.
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
    // Raw view: all 33 points, unfiltered, indexed positionally — what the
    // canvas skeleton overlay draws from.
    raw: pose.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0, visibility: p.visibility ?? 1 })),
  };
}
