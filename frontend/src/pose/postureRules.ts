// Form-violation detection with per-joint debounce — faithful port of
// stale/core/exercise/posture_rules.py, including fitness-level threshold
// widening and extra leniency for user-reported limited joints.

import type { AngleRangeDef, FitnessLevel } from '../api/types';
import { contains } from './stateMachine';

// Widened from 8/5 — the measured angle ranges these compare against are
// noisy (thin, mixed-quality training data), so require more sustained
// evidence before nagging the user about a form issue or a lost joint.
const FRAMES_TO_ALERT = 12;
const FRAMES_TO_MISSING = 8;

// Mirrors User.threshold_modifier in core/app_model.py
export const THRESHOLD_MODIFIER: Record<FitnessLevel, number> = {
  beginner: 1.30,
  intermediate: 1.00,
  advanced: 0.85,
};

// Mirrors User.LIMITATION_JOINT_MAP in core/app_model.py, but with joint
// names from this app's own angle space (angles.ts) rather than the stale
// desktop pipeline's (which used right_arm_body/left_arm_body — a joint
// this app never computes).
export const LIMITATION_JOINT_MAP: Record<string, string[]> = {
  right_knee: ['right_knee'],
  left_knee: ['left_knee'],
  lower_back: ['spine'],
  right_shoulder: ['right_shoulder'],
  left_shoulder: ['left_shoulder'],
  right_elbow: ['right_elbow'],
  left_elbow: ['left_elbow'],
};

export function limitedJointsFor(limitations: string[]): string[] {
  return limitations.flatMap(l => LIMITATION_JOINT_MAP[l] ?? []);
}

export interface PostureIssue {
  joint: string;
  severity: 'high' | 'medium' | 'low';
  message: string;
  direction: 'too_low' | 'too_high' | 'missing';
  value: number | null;
  expectedMin: number;
  expectedMax: number;
}

export class PostureRules {
  private modifier: number;
  private limitedJoints: string[];
  private counters: Record<string, number> = {};

  constructor(modifier = 1.0, limitedJoints: string[] = []) {
    this.modifier = modifier;
    this.limitedJoints = limitedJoints;
  }

  // Widens the raw Mongo-measured range around its own midpoint — beginners
  // (modifier > 1) get more slack, advanced (modifier < 1) get stricter, and
  // a self-reported limited joint gets an extra 1.4x on top of that.
  private adjustRange(joint: string, r: AngleRangeDef): AngleRangeDef {
    if (this.modifier === 1.0 && !this.limitedJoints.includes(joint)) return r;   // nothing to adjust

    const mid = (r.min + r.max) / 2;
    let half = ((r.max - r.min) / 2) * this.modifier;
    if (this.limitedJoints.includes(joint)) half *= 1.4;

    return { ...r, min: Math.round((mid - half) * 10) / 10, max: Math.round((mid + half) * 10) / 10 };
  }

  /** A required joint isn't visible this frame — flag it once the debounce
   * counter clears FRAMES_TO_MISSING (a brief occlusion isn't worth alarming
   * over). Resets to a fresh count each time the joint reappears, since the
   * caller resets `counters[joint]` to 0 the moment it's seen again. */
  private missingJointIssue(joint: string, range: AngleRangeDef): PostureIssue | null {
    this.counters[joint] = (this.counters[joint] ?? 0) + 1;
    if (this.counters[joint] < FRAMES_TO_MISSING) return null;
    return {
      joint,
      severity: 'high',
      message: `Can't see your ${joint.replace(/_/g, ' ')} — make sure it's in frame`,
      direction: 'missing',
      value: null,
      expectedMin: range.min,
      expectedMax: range.max,
    };
  }

  /** A visible joint outside its (fitness/limitation-adjusted) angle range —
   * flag it once the debounce counter clears FRAMES_TO_ALERT, using the
   * exercise's own correction copy when it has one. */
  private outOfRangeIssue(joint: string, adjusted: AngleRangeDef, value: number): PostureIssue | null {
    if (contains(adjusted, value)) {
      this.counters[joint] = 0;
      return null;
    }

    this.counters[joint] = (this.counters[joint] ?? 0) + 1;
    if (this.counters[joint] < FRAMES_TO_ALERT) return null;

    const direction = value < adjusted.min ? 'too_low' : 'too_high';
    let severity = (adjusted.corrections['severity'] ?? 'medium') as PostureIssue['severity'];
    let message = adjusted.corrections[direction] ??
      `${direction.replace('_', ' ')} (${value.toFixed(0)}°, expected ${adjusted.min.toFixed(0)}–${adjusted.max.toFixed(0)}°)`;

    if (this.limitedJoints.includes(joint)) {
      severity = 'low';
      message = `[Adapted] ${message}`;
    }

    return {
      joint,
      severity,
      message,
      direction,
      value: Math.round(value),
      expectedMin: adjusted.min,
      expectedMax: adjusted.max,
    };
  }

  analyze(
    angles: Record<string, number>,
    activeRules: Record<string, AngleRangeDef>,
  ): PostureIssue[] {
    const issues: PostureIssue[] = [];

    for (const [joint, range] of Object.entries(activeRules)) {
      const issue = joint in angles
        ? this.outOfRangeIssue(joint, this.adjustRange(joint, range), angles[joint])
        : this.missingJointIssue(joint, range);
      if (issue) issues.push(issue);
    }

    // A joint that's no longer part of the active rules (e.g. the exercise
    // moved to a phase that doesn't check it) shouldn't keep counting
    // toward a stale alert the next time it becomes relevant.
    for (const joint of Object.keys(this.counters)) {
      if (!(joint in activeRules)) this.counters[joint] = 0;
    }

    return issues;
  }

  reset(): void {
    this.counters = {};
  }
}

/** Mirrors core/user/workout_history.rep_form_score. */
export function repFormScore(errorJoints: string[]): number {
  const unique = new Set(errorJoints).size;
  return Math.max(0, Math.round((100 - unique * 15) * 10) / 10);
}
