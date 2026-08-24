// Pose landmarks → joint angles (degrees).
// Faithful port of core/ml/angles.py (the 10 normalized-2D angles the models
// and the Mongo ranges were measured in) plus the legacy spine angle from
// core/detection/angle_calculator.py (pixel-space 3D, needed for the global
// back-straightness constraint).

import type { Landmarks, LandmarkName } from './landmarks';

// (joint_a, vertex, joint_c) — angle measured at vertex. Mirrors ANGLE_DEFS.
const ANGLE_DEFS: Record<string, [LandmarkName, LandmarkName, LandmarkName]> = {
  left_knee:      ['left_hip',       'left_knee',      'left_ankle'],
  right_knee:     ['right_hip',      'right_knee',     'right_ankle'],
  left_hip:       ['left_shoulder',  'left_hip',       'left_knee'],
  right_hip:      ['right_shoulder', 'right_hip',      'right_knee'],
  left_elbow:     ['left_shoulder',  'left_elbow',     'left_wrist'],
  right_elbow:    ['right_shoulder', 'right_elbow',    'right_wrist'],
  left_shoulder:  ['left_hip',       'left_shoulder',  'left_elbow'],
  right_shoulder: ['right_hip',      'right_shoulder', 'right_elbow'],
  left_ankle:     ['left_knee',      'left_ankle',     'left_foot_index'],
  right_ankle:    ['right_knee',     'right_ankle',    'right_foot_index'],
};

const MIN_VISIBILITY = 0.4;

function angle2d(
  a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number },
): number {
  const bax = a.x - b.x, bay = a.y - b.y;
  const bcx = c.x - b.x, bcy = c.y - b.y;
  const norm = Math.hypot(bax, bay) * Math.hypot(bcx, bcy);
  if (norm < 1e-9) return 0;
  const cos = Math.min(1, Math.max(-1, (bax * bcx + bay * bcy) / norm));
  return (Math.acos(cos) * 180) / Math.PI;
}

function angle3d(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number },
  c: { x: number; y: number; z: number },
): number {
  const ba = [a.x - b.x, a.y - b.y, a.z - b.z];
  const bc = [c.x - b.x, c.y - b.y, c.z - b.z];
  const norm = Math.hypot(...ba) * Math.hypot(...bc);
  if (norm < 1e-9) return 0;
  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const cos = Math.min(1, Math.max(-1, dot / norm));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * All joint angles visible in this frame, keyed by angle name.
 * `width`/`height` are the video dimensions — only the spine angle needs
 * them (it replicates the legacy pixel-space computation exactly).
 */
export function computeAngles(lms: Landmarks, width: number, height: number): Record<string, number> {
  const out: Record<string, number> = {};

  for (const [name, [a, b, c]] of Object.entries(ANGLE_DEFS)) {
    const pa = lms[a], pb = lms[b], pc = lms[c];
    if (!pa || !pb || !pc) continue;
    if (pa.visibility < MIN_VISIBILITY || pb.visibility < MIN_VISIBILITY ||
        pc.visibility < MIN_VISIBILITY) continue;
    out[name] = angle2d(pa, pb, pc);
  }

  // Spine: angle at the hip between the shoulder and a straight-up reference
  // point 50px above the hip — computed in pixel space with z scaled by width,
  // exactly like AngleCalculator (the seed spine range was set in that space).
  for (const side of ['right', 'left'] as const) {
    const shoulder = lms[`${side}_shoulder`];
    const hip = lms[`${side}_hip`];
    if (!shoulder || !hip) continue;
    if (shoulder.visibility < MIN_VISIBILITY || hip.visibility < MIN_VISIBILITY) continue;
    const sPx = { x: shoulder.x * width, y: shoulder.y * height, z: shoulder.z * width };
    const hPx = { x: hip.x * width, y: hip.y * height, z: hip.z * width };
    const vertical = { x: hPx.x, y: hPx.y - 50, z: hPx.z };
    out.spine = angle3d(sPx, hPx, vertical);
    break;
  }

  return out;
}
