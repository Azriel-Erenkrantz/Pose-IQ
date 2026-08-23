"""
Tests for core/ml/ — phase classifier, feature engineering, smoothing, reps.

No camera, no MediaPipe. All tests use synthetic angle dicts.
Violation-detector tests moved to stale/tests/test_violations.py along with
core/ml/violations.py (unused anywhere else — not even by the desktop
pipeline — but split out here since it shared this file with classifier.py,
which is still used by core/ml/eval.py).

Run from project root:
    python -m pytest tests/test_ml.py -v
"""
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np

from core.exercise.exercise_model import AngleRange, Exercise, Phase
from core.ml.classifier import (
    PhaseScore, _boundary_score, _gaussian_score, _model_cache,
    all_scores, classify, classify_trained, score_phase,
)


# ---------------------------------------------------------------------------
# Shared fake exercise (same shape as a real squat)
# ---------------------------------------------------------------------------

def _make_exercise() -> Exercise:
    def _ar(lo, hi, mean=None, std=None, too_low='', too_high='', sev='medium'):
        c = {}
        if too_low: c['too_low'] = too_low
        if too_high: c['too_high'] = too_high
        c['severity'] = sev
        return AngleRange(lo, hi, corrections=c, mean=mean, std=std)

    return Exercise(
        id='squat', name='Squat', description='', muscle_groups=['quads'],
        primary_joints=['right_knee', 'left_knee'],
        mandatory_start_joints=['right_knee', 'left_knee'],
        global_constraints={
            'spine': _ar(0, 30, too_high='Keep back straight', sev='high'),
        },
        phases=[
            Phase(
                name='standing', order=1, is_initial=True,
                instruction='Stand straight',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='stable',
                angles={
                    'right_knee': _ar(155, 180, mean=170.0, std=6.0,
                                      too_low='Straighten knees before starting'),
                    'left_knee':  _ar(155, 180, mean=170.0, std=6.0,
                                      too_low='Straighten knees before starting'),
                },
            ),
            Phase(
                name='descending', order=2, is_initial=False,
                instruction='Squat down',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='decreasing',
                angles={
                    'right_knee': _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Too deep on the way down',
                                      too_high='Bend more as you descend', sev='high'),
                    'left_knee':  _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Too deep on the way down',
                                      too_high='Bend more as you descend', sev='high'),
                },
            ),
            Phase(
                name='hold', order=3, is_initial=False,
                instruction='Hold squat',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='stable',
                angles={
                    'right_knee': _ar(70, 95, too_high='Go lower', too_low="Don't go so deep", sev='high'),
                    'left_knee':  _ar(70, 95, too_high='Go lower', too_low="Don't go so deep", sev='high'),
                },
            ),
            Phase(
                name='ascending', order=4, is_initial=False,
                instruction='Rise up',
                diagnostic_joints=['right_knee', 'left_knee'],
                motion_direction='increasing',
                angles={
                    'right_knee': _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Keep pushing up', sev='high'),
                    'left_knee':  _ar(90, 155, mean=120.0, std=18.0,
                                      too_low='Keep pushing up', sev='high'),
                },
            ),
        ],
    )


STANDING  = {'right_knee': 170.0, 'left_knee': 170.0}
DESCEND   = {'right_knee': 120.0, 'left_knee': 120.0}
HOLD      = {'right_knee': 82.0,  'left_knee': 82.0}
ASCEND    = {'right_knee': 120.0, 'left_knee': 120.0}


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

class TestScoringFunctions(unittest.TestCase):

    def test_gaussian_peaks_at_mean(self):
        self.assertAlmostEqual(_gaussian_score(120.0, 120.0, 18.0), 1.0)

    def test_gaussian_decays_away_from_mean(self):
        self.assertLess(_gaussian_score(160.0, 120.0, 18.0),
                        _gaussian_score(130.0, 120.0, 18.0))

    def test_gaussian_always_between_0_and_1(self):
        for v in [0, 50, 100, 150, 200]:
            s = _gaussian_score(v, 120.0, 18.0)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_boundary_inside_is_1(self):
        self.assertEqual(_boundary_score(120.0, 90.0, 155.0), 1.0)

    def test_boundary_at_edge_is_1(self):
        self.assertEqual(_boundary_score(90.0, 90.0, 155.0), 1.0)
        self.assertEqual(_boundary_score(155.0, 90.0, 155.0), 1.0)

    def test_boundary_outside_decays(self):
        s = _boundary_score(200.0, 90.0, 155.0)
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1.0)

    def test_boundary_far_outside_reaches_0(self):
        # 400° is very far outside [90, 155] (span=65, so 2×span away = 220°)
        self.assertEqual(_boundary_score(400.0, 90.0, 155.0), 0.0)


# ---------------------------------------------------------------------------
# Phase scoring
# ---------------------------------------------------------------------------

class TestPhaseScore(unittest.TestCase):

    def setUp(self):
        self.ex = _make_exercise()

    def test_standing_angles_score_high_for_standing_phase(self):
        standing_phase = self.ex.get_phase('standing')
        result = score_phase(standing_phase, STANDING)
        self.assertGreater(result.score, 0.8)

    def test_hold_angles_score_low_for_standing_phase(self):
        standing_phase = self.ex.get_phase('standing')
        result = score_phase(standing_phase, HOLD)
        self.assertLess(result.score, 0.2)

    def test_missing_diagnostic_joint_lowers_score(self):
        phase = self.ex.get_phase('standing')
        full_result    = score_phase(phase, STANDING)
        partial_result = score_phase(phase, {'right_knee': 170.0})  # left missing
        self.assertLess(partial_result.score, full_result.score)

    def test_matched_joints_counted_correctly(self):
        phase = self.ex.get_phase('standing')
        result = score_phase(phase, STANDING)
        self.assertEqual(result.matched_joints, 2)
        self.assertEqual(result.total_joints, 2)

    def test_phase_with_no_diagnostic_joints_scores_zero(self):
        bare_phase = Phase(name='empty', order=99, angles={}, diagnostic_joints=[])
        result = score_phase(bare_phase, STANDING)
        self.assertEqual(result.score, 0.0)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.ex = _make_exercise()

    def test_standing_classified_as_standing(self):
        phase = classify(self.ex, STANDING)
        self.assertEqual(phase.name, 'standing')

    def test_hold_classified_as_hold(self):
        phase = classify(self.ex, HOLD)
        self.assertEqual(phase.name, 'hold')

    def test_descending_classified_as_descending_or_ascending(self):
        # Both descending and ascending have identical angle ranges;
        # without motion direction both are valid answers
        phase = classify(self.ex, DESCEND)
        self.assertIn(phase.name, ('descending', 'ascending'))

    def test_motion_direction_breaks_tie_descending(self):
        # If knee angle is decreasing → descending
        phase = classify(self.ex, DESCEND,
                         prev_primary_angle=140.0,
                         current_primary_angle=120.0)
        self.assertEqual(phase.name, 'descending')

    def test_motion_direction_breaks_tie_ascending(self):
        # If knee angle is increasing → ascending
        phase = classify(self.ex, ASCEND,
                         prev_primary_angle=100.0,
                         current_primary_angle=120.0)
        self.assertEqual(phase.name, 'ascending')

    def test_all_scores_sorted_best_first(self):
        scores = all_scores(self.ex, HOLD)
        values = [s.score for s in scores]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_all_scores_has_entry_for_every_phase(self):
        scores = all_scores(self.ex, HOLD)
        self.assertEqual(len(scores), len(self.ex.phases))

    def test_empty_angles_returns_a_phase(self):
        # Should not crash; returns whichever phase scores highest on nothing
        phase = classify(self.ex, {})
        self.assertIsNotNone(phase)

    def test_returns_none_for_exercise_with_no_phases(self):
        empty_ex = Exercise('x', 'X', '', [], [])
        self.assertIsNone(classify(empty_ex, STANDING))


# ---------------------------------------------------------------------------
# Feature vector format
# ---------------------------------------------------------------------------

class TestFeatureVector(unittest.TestCase):
    """
    The 24-element feature vector (12 angles + 12 per-joint velocities) must be
    built identically during training (collect_samples) and inference
    (classify_trained). If they diverge, the RF silently produces garbage.
    These tests lock down the format.
    """

    def test_length_is_12_angles_plus_12_deltas(self):
        from core.ml.trainer import JOINTS, N_FEATURES, angles_to_features
        feat = angles_to_features({})
        self.assertEqual(len(feat), 2 * len(JOINTS))
        self.assertEqual(len(feat), N_FEATURES)
        self.assertEqual(len(feat), 24)

    def test_missing_joints_are_minus_one(self):
        from core.ml.trainer import angles_to_features
        feat = angles_to_features({})
        # All 12 angle slots should be -1.0
        self.assertTrue(all(v == -1.0 for v in feat[:12]))

    def test_missing_deltas_are_zero(self):
        from core.ml.trainer import angles_to_features
        feat = angles_to_features({'right_knee': 90.0})
        self.assertTrue(all(v == 0.0 for v in feat[12:]))

    def test_present_joint_value_preserved(self):
        from core.ml.trainer import JOINTS, angles_to_features
        feat = angles_to_features({'right_knee': 90.0})
        self.assertEqual(feat[JOINTS.index('right_knee')], 90.0)

    def test_absent_joint_is_minus_one_not_zero(self):
        from core.ml.trainer import JOINTS, angles_to_features
        feat = angles_to_features({'right_knee': 90.0})
        self.assertEqual(feat[JOINTS.index('left_knee')], -1.0)

    def test_delta_slot_matches_joint_order(self):
        from core.ml.trainer import JOINTS, angles_to_features
        feat = angles_to_features({}, deltas={'right_knee': 7.5})
        self.assertAlmostEqual(feat[12 + JOINTS.index('right_knee')], 7.5, places=4)
        # Other delta slots stay 0
        self.assertEqual(feat[12 + JOINTS.index('left_knee')], 0.0)

    def test_negative_delta_preserved(self):
        from core.ml.trainer import JOINTS, angles_to_features
        feat = angles_to_features({}, deltas={'left_hip': -4.2})
        self.assertAlmostEqual(feat[12 + JOINTS.index('left_hip')], -4.2, places=4)

    def test_feature_dtype_is_float32(self):
        from core.ml.trainer import angles_to_features
        feat = angles_to_features({'right_knee': 90.0}, deltas={'right_knee': 1.0})
        self.assertEqual(feat.dtype, np.float32)

    def test_joints_order_matches_trainer_constant(self):
        """The first 12 features must be in the same order as trainer.JOINTS."""
        from core.ml.trainer import JOINTS, angles_to_features
        angles = {j: float(i * 10) for i, j in enumerate(JOINTS)}
        feat = angles_to_features(angles)
        for i, joint in enumerate(JOINTS):
            self.assertAlmostEqual(feat[i], angles[joint],
                msg=f'Feature index {i} should be {joint}={angles[joint]}, got {feat[i]}')


class TestRollingDeltas(unittest.TestCase):
    """trainer.rolling_deltas (offline) and classifier.DeltaTracker (live)
    must produce identical numbers for the same frame stream."""

    FRAMES = [
        {'right_knee': 170.0, 'left_knee': 168.0},
        {'right_knee': 160.0, 'left_knee': 162.0},
        {'right_knee': 148.0, 'left_knee': 150.0},
        {'right_knee': 140.0},                       # left knee occluded
        {'right_knee': 133.0, 'left_knee': 130.0},
    ]

    def test_offline_deltas_mean_of_pairs(self):
        from core.ml.trainer import rolling_deltas
        d = rolling_deltas(self.FRAMES, 2)
        self.assertAlmostEqual(d['right_knee'], ((160 - 170) + (148 - 160)) / 2)
        self.assertAlmostEqual(d['left_knee'], ((162 - 168) + (150 - 162)) / 2)

    def test_first_frame_has_no_deltas(self):
        from core.ml.trainer import rolling_deltas
        self.assertEqual(rolling_deltas(self.FRAMES, 0), {})

    def test_occluded_joint_pairs_skipped(self):
        from core.ml.trainer import rolling_deltas
        # frame 3 lost left_knee → pairs (2,3) and (3,4) contribute nothing for it
        d = rolling_deltas(self.FRAMES, 4, window=2)
        self.assertIn('right_knee', d)
        self.assertNotIn('left_knee', d)

    def test_live_tracker_matches_offline(self):
        """A 30fps live stream must produce the same °/sec as offline rate=30."""
        from core.ml.classifier import DeltaTracker
        from core.ml.trainer import rolling_deltas
        fps = 30.0
        tracker = DeltaTracker(window=5)
        for i, frame in enumerate(self.FRAMES):
            live = tracker.update(frame, now=i / fps)
            offline = rolling_deltas(self.FRAMES, i, window=5, rate=fps)
            self.assertEqual(set(live), set(offline), f'frame {i}')
            for j in offline:
                self.assertAlmostEqual(live[j], offline[j], places=4,
                                       msg=f'frame {i}, joint {j}')

    def test_rate_scales_to_degrees_per_second(self):
        from core.ml.trainer import rolling_deltas
        per_frame = rolling_deltas(self.FRAMES, 2)
        per_sec   = rolling_deltas(self.FRAMES, 2, rate=30.0)
        for j in per_frame:
            self.assertAlmostEqual(per_sec[j], per_frame[j] * 30.0)


class TestPhaseSmoother(unittest.TestCase):
    """Majority vote + legal-cycle constraint over raw phase predictions."""

    def setUp(self):
        from core.ml.smoother import PhaseSmoother
        self.smoother = PhaseSmoother(_make_exercise(), window=5, resync_after=6)

    def _feed(self, names):
        out = []
        for n in names:
            out.append(self.smoother.update(n))
        return out

    def test_single_frame_flicker_erased(self):
        out = self._feed(['standing'] * 4 + ['descending'] + ['standing'] * 4)
        self.assertTrue(all(p == 'standing' for p in out))

    def test_sustained_transition_accepted(self):
        out = self._feed(['standing'] * 5 + ['descending'] * 5)
        self.assertEqual(out[-1], 'descending')

    def test_illegal_jump_rejected(self):
        # standing → ascending skips descending+hold — must be held back
        out = self._feed(['standing'] * 5 + ['ascending'] * 3)
        self.assertEqual(out[-1], 'standing')

    def test_persistent_illegal_winner_resyncs(self):
        # ...but if the model keeps insisting, resync instead of deadlocking
        out = self._feed(['standing'] * 5 + ['ascending'] * 12)
        self.assertEqual(out[-1], 'ascending')

    def test_none_prediction_keeps_current(self):
        self._feed(['standing'] * 5)
        self.assertEqual(self.smoother.update(None), 'standing')

    def test_reset_clears_state(self):
        self._feed(['standing'] * 5)
        self.smoother.reset()
        self.assertIsNone(self.smoother.current)


# ---------------------------------------------------------------------------
# Real trained models (skipped if data/models/ not populated)
# ---------------------------------------------------------------------------

SQUAT_MODEL_PATH = Path('data/models/phase_squat.joblib')


class TestRealTrainedModel(unittest.TestCase):
    """
    Tests against the actual squat model produced by trainer.py.
    Skipped when the file doesn't exist.

    These tests catch the bugs that the synthetic-model tests cannot:
      - Feature shape mismatch between training and inference
      - Phase name drift (model class no longer matches exercise phase)
      - Trainer output missing required keys
    """

    @classmethod
    def setUpClass(cls):
        if not SQUAT_MODEL_PATH.exists():
            raise unittest.SkipTest(
                f'No model at {SQUAT_MODEL_PATH} — run: python -m core.ml.trainer'
            )
        import joblib
        cls.bundle = joblib.load(SQUAT_MODEL_PATH)
        cls.ex = _make_exercise()

    def test_bundle_has_all_required_keys(self):
        for key in ('model', 'classes', 'joints', 'exercise_id',
                    'cv_accuracy', 'n_samples', 'class_counts'):
            self.assertIn(key, self.bundle, f'bundle missing key: {key}')

    def test_feature_shape_matches_inference(self):
        """
        The most important test: the number of features the model expects must
        equal what classify_trained actually builds for its feature_version.
        If angles_to_features changes but the model is not retrained, this fails.
        """
        from core.ml.trainer import JOINTS, N_FEATURES
        version = self.bundle.get('feature_version', 1)
        expected = N_FEATURES if version >= 2 else len(JOINTS) + 1
        actual = self.bundle['model'].n_features_in_
        self.assertEqual(actual, expected,
            f'Saved model (feature_version={version}) expects {actual} features, '
            f'inference builds {expected}. Retrain.')

    def test_all_classes_are_valid_phase_names(self):
        """Every class the model learned must correspond to a real phase."""
        for cls in self.bundle['classes']:
            phase = self.ex.get_phase(str(cls))
            self.assertIsNotNone(phase,
                f'Model class "{cls}" has no matching phase in the exercise')

    def test_exercise_id_is_squat(self):
        self.assertEqual(self.bundle['exercise_id'], 'squat')

    def test_accuracy_stored_and_positive(self):
        acc = self.bundle['cv_accuracy']
        # accuracy can be None for very small datasets, otherwise must be in [0,1]
        if acc is not None:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_n_samples_positive(self):
        self.assertGreater(self.bundle['n_samples'], 0)

    def _features(self, angles, deltas=None):
        """Build features matching the loaded bundle's feature_version."""
        from core.ml.trainer import JOINTS, angles_to_features
        if self.bundle.get('feature_version', 1) >= 2:
            return angles_to_features(angles, deltas).reshape(1, -1)
        # Legacy 13-feature layout: single primary delta as last element
        primary = max(deltas, key=lambda j: abs(deltas[j])) if deltas else None
        delta = deltas[primary] if primary else 0.0
        return np.array(
            [[angles.get(j, -1.0) for j in JOINTS] + [delta]], dtype=np.float32)

    def test_predict_does_not_crash_on_all_missing(self):
        """All -1.0 features (no joints visible) must not raise."""
        feat = self._features({})
        pred = self.bundle['model'].predict(feat)
        self.assertEqual(len(pred), 1)
        self.assertIn(str(pred[0]), [str(c) for c in self.bundle['classes']])

    def test_predict_on_standing_angles_returns_valid_phase(self):
        """Realistic standing angles must produce a valid phase name without crashing."""
        standing_full = {
            'right_knee': 167.0, 'left_knee': 157.0,
            'right_hip':  167.0, 'left_hip':  157.0,
            'right_ankle': 124.0, 'left_ankle': 95.0,
        }
        feat = self._features(standing_full, {'right_knee': -0.5, 'left_knee': -0.5})
        pred = str(self.bundle['model'].predict(feat)[0])
        valid = [str(c) for c in self.bundle['classes']]
        self.assertIn(pred, valid, f'Prediction "{pred}" is not a known phase')

    def test_negative_delta_not_classified_as_standing(self):
        """A strongly decreasing angle should never be classified as standing."""
        mid = {'right_knee': 130.0, 'left_knee': 130.0}
        feat = self._features(mid, {'right_knee': -10.0, 'left_knee': -10.0})
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertNotEqual(pred, 'standing',
            'Angle at 130° and sharply decreasing should not be standing')

    def test_positive_delta_not_classified_as_standing(self):
        """A strongly increasing angle from a low position should not be standing."""
        mid = {'right_knee': 110.0, 'left_knee': 110.0}
        feat = self._features(mid, {'right_knee': +10.0, 'left_knee': +10.0})
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertNotEqual(pred, 'standing')

    def test_classify_trained_returns_phase_and_method(self):
        """End-to-end: classify_trained returns (Phase, str) — not None."""
        phase, method = classify_trained(self.ex, STANDING,
                                         prev_primary_angle=170.0,
                                         current_primary_angle=170.0)
        self.assertIsNotNone(phase)
        self.assertIsInstance(phase.name, str)
        self.assertIn(method, ('rf', 'gaussian'))

    def test_classify_trained_uses_rf_for_squat(self):
        """The real model file exists, so method must be 'rf', not 'gaussian'."""
        _model_cache.clear()
        phase, method = classify_trained(self.ex, STANDING,
                                          prev_primary_angle=170.0,
                                          current_primary_angle=170.0)
        _model_cache.clear()
        self.assertEqual(method, 'rf')

    def test_negative_delta_predicts_descending(self):
        """
        Mid-range angle with strongly negative velocities (angles falling fast)
        should predict 'descending'. Requires hip angles — the model was
        trained with multi-joint vectors, not just knees.
        """
        mid = {'right_knee': 120.0, 'left_knee': 120.0,
               'right_hip': 130.0, 'left_hip': 130.0}
        down = {j: -15.0 for j in mid}
        feat = self._features(mid, down)
        pred = str(self.bundle['model'].predict(feat)[0])
        self.assertEqual(pred, 'descending',
            f'Mid-range angle with velocity -15°/frame classified as "{pred}" — '
            'the velocity features should signal descent')

    def test_delta_feature_is_used_by_model(self):
        """
        The same mid-range angle with opposite velocities must not produce
        identical predictions — confirms the velocity features actually
        influence the RF.
        """
        mid = {'right_knee': 120.0, 'left_knee': 120.0,
               'right_hip': 130.0, 'left_hip': 130.0}
        feat_neg = self._features(mid, {j: -15.0 for j in mid})
        feat_pos = self._features(mid, {j: +15.0 for j in mid})
        pred_neg = str(self.bundle['model'].predict(feat_neg)[0])
        pred_pos = str(self.bundle['model'].predict(feat_pos)[0])
        self.assertNotEqual(pred_neg, pred_pos,
            'Opposite velocities at the same angle produce identical predictions — '
            'the velocity features are not influencing the model')


# ---------------------------------------------------------------------------
# Random Forest: test with a tiny synthetic model
# ---------------------------------------------------------------------------

class TestClassifyTrainedRF(unittest.TestCase):
    """
    Train a minimal Random Forest inline (no videos, no MongoDB) and verify
    that classify_trained uses it and returns the correct phase.

    This is how you test the RF path before real reference videos exist.
    """

    def setUp(self):
        self.ex = _make_exercise()

    def _build_bundle(self) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from core.ml.trainer import JOINTS

        # 10 examples of each phase — enough to train a trivial classifier
        # Features: 12 joint angles + 1 angle_delta (last element)
        samples = {
            'standing':   [170.0, 170.0] + [-1.0] * 10 + [0.0],   # stable
            'descending': [125.0, 125.0] + [-1.0] * 10 + [-5.0],  # angle decreasing
            'hold':       [82.0,  82.0]  + [-1.0] * 10 + [0.0],   # stable at bottom
            'ascending':  [110.0, 110.0] + [-1.0] * 10 + [+5.0],  # angle increasing
        }
        X, y = [], []
        for label, feat in samples.items():
            for _ in range(20):
                X.append(feat)
                y.append(label)

        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(np.array(X, dtype=np.float32), y)

        return {
            'model':       clf,
            'classes':     list(clf.classes_),
            'joints':      JOINTS,
            'exercise_id': 'squat',
        }

    def test_rf_classifies_standing(self):
        import joblib
        bundle = self._build_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'phase_squat.joblib'
            joblib.dump(bundle, model_path)
            _model_cache.clear()
            with patch('core.ml.classifier.MODELS_DIR', Path(tmpdir)):
                phase, method = classify_trained(self.ex, STANDING)
            _model_cache.clear()
        self.assertEqual(method, 'rf')
        self.assertEqual(phase.name, 'standing')

    def test_rf_classifies_hold(self):
        import joblib
        bundle = self._build_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'phase_squat.joblib'
            joblib.dump(bundle, model_path)
            _model_cache.clear()
            with patch('core.ml.classifier.MODELS_DIR', Path(tmpdir)):
                phase, method = classify_trained(self.ex, HOLD)
            _model_cache.clear()
        self.assertEqual(method, 'rf')
        self.assertEqual(phase.name, 'hold')

    def test_rf_falls_back_to_gaussian_when_no_model(self):
        _model_cache.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('core.ml.classifier.MODELS_DIR', Path(tmpdir)):
                phase, method = classify_trained(self.ex, STANDING)
        _model_cache.clear()
        self.assertEqual(method, 'gaussian')
        self.assertEqual(phase.name, 'standing')

    def test_rf_method_reported_correctly(self):
        import joblib
        bundle = self._build_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump(bundle, Path(tmpdir) / 'phase_squat.joblib')
            _model_cache.clear()
            with patch('core.ml.classifier.MODELS_DIR', Path(tmpdir)):
                _, method = classify_trained(self.ex, HOLD)
            _model_cache.clear()
        self.assertIn(method, ('rf', 'gaussian'))


# ---------------------------------------------------------------------------
# Rep-level metrics (core/ml/reps.py)
# ---------------------------------------------------------------------------

# Squat cycle as sorted by phase order in _make_exercise()
CYCLE = ['standing', 'descending', 'hold', 'ascending']


def _seq(*runs):
    """Expand ('standing', 3), ('descending', 4), ... into a frame sequence."""
    out = []
    for phase, n in runs:
        out.extend([phase] * n)
    return out


class TestExtractReps(unittest.TestCase):

    def test_full_cycle_is_one_rep(self):
        from core.ml.reps import extract_reps
        seq = _seq(('standing', 5), ('descending', 5), ('hold', 3),
                   ('ascending', 5), ('standing', 5))
        reps = extract_reps(seq, CYCLE)
        self.assertEqual(len(reps), 1)

    def test_anchor_is_entry_into_final_phase(self):
        from core.ml.reps import extract_reps
        seq = _seq(('standing', 5), ('descending', 5), ('hold', 3),
                   ('ascending', 5), ('standing', 5))
        rep = extract_reps(seq, CYCLE)[0]
        self.assertEqual(rep.anchor_idx, 13)   # 5 standing + 5 desc + 3 hold
        self.assertEqual(rep.start_idx, 5)     # descending began here
        self.assertEqual(rep.end_idx, 17)      # last ascending frame

    def test_two_reps_counted(self):
        from core.ml.reps import extract_reps
        one = [('descending', 4), ('hold', 2), ('ascending', 4), ('standing', 3)]
        seq = _seq(('standing', 3), *one, *one)
        self.assertEqual(len(extract_reps(seq, CYCLE)), 2)

    def test_skipped_hold_still_a_rep(self):
        # Fast reps often have no visible hold — descending straight into
        # ascending must still count.
        from core.ml.reps import extract_reps
        seq = _seq(('standing', 4), ('descending', 5), ('ascending', 5),
                   ('standing', 4))
        self.assertEqual(len(extract_reps(seq, CYCLE)), 1)

    def test_ascending_without_descending_is_not_a_rep(self):
        # Isolated final-phase blip (label artifact or model flicker)
        from core.ml.reps import extract_reps
        seq = _seq(('standing', 5), ('ascending', 4), ('standing', 5))
        self.assertEqual(len(extract_reps(seq, CYCLE)), 0)

    def test_unknown_frames_do_not_split_a_segment(self):
        # Truth labels have gaps; None/'unknown' frames carry no information
        from core.ml.reps import extract_reps
        seq = (_seq(('standing', 3), ('descending', 3)) + [None, 'unknown'] +
               _seq(('descending', 2), ('ascending', 4), ('standing', 3)))
        self.assertEqual(len(extract_reps(seq, CYCLE)), 1)

    def test_flicker_produces_extra_reps(self):
        # Raw (unsmoothed) desc/asc flicker: each desc→asc counts — this is
        # exactly the double-counting the metric must expose.
        from core.ml.reps import extract_reps
        seq = _seq(('standing', 3), ('descending', 2), ('ascending', 2),
                   ('descending', 2), ('ascending', 2), ('standing', 3))
        self.assertEqual(len(extract_reps(seq, CYCLE)), 2)

    def test_empty_sequence(self):
        from core.ml.reps import extract_reps
        self.assertEqual(extract_reps([], CYCLE), [])


class TestMatchReps(unittest.TestCase):

    def test_exact_anchors_all_match(self):
        from core.ml.reps import match_reps
        pairs = match_reps([10, 50, 90], [10, 50, 90], tol_frames=5)
        self.assertEqual(len(pairs), 3)

    def test_offset_within_tolerance_matches(self):
        from core.ml.reps import match_reps
        pairs = match_reps([10], [13], tol_frames=5)
        self.assertEqual(pairs, [(0, 0)])

    def test_offset_beyond_tolerance_no_match(self):
        from core.ml.reps import match_reps
        self.assertEqual(match_reps([10], [20], tol_frames=5), [])

    def test_each_anchor_used_once(self):
        # Two predictions near one truth rep: only the closer one matches
        from core.ml.reps import match_reps
        pairs = match_reps([10], [9, 13], tol_frames=5)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (0, 0))   # 9 is closer than 13

    def test_greedy_prefers_smallest_offset(self):
        from core.ml.reps import match_reps
        # truth 10 could match pred 14 (|4|), but pred 11 (|1|) wins it;
        # pred 14 then matches truth 15 (|1|)
        pairs = sorted(match_reps([10, 15], [11, 14], tol_frames=5))
        self.assertEqual(pairs, [(0, 0), (1, 1)])


class TestRepMetrics(unittest.TestCase):

    REP = [('descending', 4), ('hold', 2), ('ascending', 4), ('standing', 4)]

    def test_perfect_prediction(self):
        from core.ml.reps import rep_metrics
        seq = _seq(('standing', 4), *self.REP, *self.REP)
        m = rep_metrics(seq, list(seq), CYCLE, tol_frames=5)
        self.assertEqual(m['truth_reps'], 2)
        self.assertEqual(m['matched'], 2)
        self.assertEqual(m['missed'], 0)
        self.assertEqual(m['extra'], 0)
        self.assertEqual(m['recall'], 1.0)
        self.assertEqual(m['precision'], 1.0)
        self.assertEqual(m['f1'], 1.0)
        self.assertEqual(m['mean_anchor_offset_frames'], 0.0)

    def test_missed_rep_lowers_recall(self):
        from core.ml.reps import rep_metrics
        truth = _seq(('standing', 4), *self.REP, *self.REP)
        # Model saw only the first rep, then flat standing
        pred = _seq(('standing', 4), *self.REP) + ['standing'] * 14
        m = rep_metrics(truth, pred, CYCLE, tol_frames=5)
        self.assertEqual(m['truth_reps'], 2)
        self.assertEqual(m['matched'], 1)
        self.assertEqual(m['missed'], 1)
        self.assertEqual(m['recall'], 0.5)

    def test_extra_rep_lowers_precision(self):
        from core.ml.reps import rep_metrics
        truth = _seq(('standing', 4), *self.REP) + ['standing'] * 14
        pred = _seq(('standing', 4), *self.REP, *self.REP)
        m = rep_metrics(truth, pred, CYCLE, tol_frames=5)
        self.assertEqual(m['truth_reps'], 1)
        self.assertEqual(m['extra'], 1)
        self.assertEqual(m['precision'], 0.5)
        self.assertEqual(m['recall'], 1.0)

    def test_no_reps_anywhere_is_safe(self):
        from core.ml.reps import rep_metrics
        seq = ['standing'] * 10
        m = rep_metrics(seq, list(seq), CYCLE, tol_frames=5)
        self.assertEqual(m['truth_reps'], 0)
        self.assertEqual(m['recall'], 0.0)
        self.assertEqual(m['f1'], 0.0)
        self.assertIsNone(m['mean_anchor_offset_frames'])


if __name__ == '__main__':
    unittest.main()
