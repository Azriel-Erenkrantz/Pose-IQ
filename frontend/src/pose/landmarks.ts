// MediaPipe Pose landmark indices (33-point model) — the subset the app uses.
// Mirrors core/ml/extractor.LANDMARK_NAMES and core/pipeline skeleton maps.

export const LM = {
  nose: 0,
  left_shoulder: 11, right_shoulder: 12,
  left_elbow: 13,    right_elbow: 14,
  left_wrist: 15,    right_wrist: 16,
  left_hip: 23,      right_hip: 24,
  left_knee: 25,     right_knee: 26,
  left_ankle: 27,    right_ankle: 28,
  left_foot_index: 31, right_foot_index: 32,
} as const;

export type LandmarkName = keyof typeof LM;

export interface Point {
  x: number;          // normalized [0,1]
  y: number;          // normalized [0,1]
  z: number;
  visibility: number;
}

export type Landmarks = Partial<Record<LandmarkName, Point>>;

/** Skeleton segments to draw, as landmark index pairs (mirrors stale/core/pipeline.py). */
export const SKELETON_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
];

/** Violating joint → skeleton segments to paint red (mirrors stale/core/pipeline.py). */
export const ERROR_TO_LINES: Record<string, [number, number][]> = {
  spine:          [[11, 23], [12, 24], [23, 24]],
  right_knee:     [[24, 26], [26, 28]],
  left_knee:      [[23, 25], [25, 27]],
  right_hip:      [[12, 24], [24, 26]],
  left_hip:       [[11, 23], [23, 25]],
  right_ankle:    [[26, 28]],
  left_ankle:     [[25, 27]],
  right_shoulder: [[12, 14], [12, 24]],
  left_shoulder:  [[11, 13], [11, 23]],
  right_elbow:    [[12, 14], [14, 16]],
  left_elbow:     [[11, 13], [13, 15]],
};

/** Joint (angle name) → landmark index where its value is anchored on screen. */
export const JOINT_ANCHOR: Record<string, number> = {
  right_knee: 26, left_knee: 25,
  right_hip: 24,  left_hip: 23,
  right_ankle: 28, left_ankle: 27,
  right_elbow: 14, left_elbow: 13,
  right_shoulder: 12, left_shoulder: 11,
  spine: 24,
};
