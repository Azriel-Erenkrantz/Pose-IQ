// Pose landmarks → joint angles (degrees).
// Faithful port of core/ml/angles.py — all 11 angles (10 joints + spine) are
// normalized-2D, the space the models and the Mongo ranges are measured in.

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

// Spine needs a stricter bar than the 3-point joint angles above: it's
// computed from just two landmarks (shoulder+hip) against a synthetic
// "straight up" reference point, with no third real landmark to anchor it —
// far more sensitive to a landmark's *position* being unreliable even while
// its visibility score still clears 0.4. Confirmed live 2026-08-27: as a
// user nears the frame edge, one side's shoulder/hip stayed just above 0.4
// while its estimated position was already distorted by perspective, so
// `spine` spiked and false-triggered the (global, high-severity, spoken)
// "don't arch your back" correction on every frame exit, not on real
// posture. Applies only to spine's own two landmarks, not the shared
// MIN_VISIBILITY gate other angles use.
const MIN_VISIBILITY_SPINE = 0.6;

// How far "up" (normalized image-space y) the synthetic vertical reference
// point sits above the hip. Doesn't affect the resulting angle at all —
// angle2d only uses vector *direction*, not length — just needs to be a
// small positive number. Mirrors core/ml/angles.py exactly.
const SPINE_UP_DELTA = 0.1;

// Angle at b in the A-B-C triangle, using x/y only (normalized 0-1 screen
// space) — the space all 10 joint angles below are measured in.
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

/** All joint angles visible in this frame, keyed by angle name. */
export function computeAngles(lms: Landmarks): Record<string, number> {
  const out: Record<string, number> = {};

  for (const [name, [a, b, c]] of Object.entries(ANGLE_DEFS)) {
    const pa = lms[a], pb = lms[b], pc = lms[c];
    if (!pa || !pb || !pc) continue;   // joint off-frame this tick — just skip it
    if (pa.visibility < MIN_VISIBILITY || pb.visibility < MIN_VISIBILITY ||
        pc.visibility < MIN_VISIBILITY) continue;   // MediaPipe isn't confident enough
    out[name] = angle2d(pa, pb, pc);
  }

  // Spine: angle at the hip between the shoulder and a synthetic straight-up
  // reference point — normalized 2D like everything else above (changed
  // 2026-08-30 from pixel-space 3D against a fixed-50px reference; see
  // core/ml/angles.py's mirror of this for why: that version could never be
  // measured from training data at all, since the training pipeline only
  // ever sees normalized landmark coordinates, no pixel dimensions).
  for (const side of ['right', 'left'] as const) {
    const shoulder = lms[`${side}_shoulder`];
    const hip = lms[`${side}_hip`];
    if (!shoulder || !hip) continue;
    if (shoulder.visibility < MIN_VISIBILITY_SPINE || hip.visibility < MIN_VISIBILITY_SPINE) continue;
    const vertical = { x: hip.x, y: hip.y - SPINE_UP_DELTA };
    out.spine = angle2d(shoulder, hip, vertical);
    break;
  }

  return out;
}
