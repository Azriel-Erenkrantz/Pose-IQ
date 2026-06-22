"""
Step 5 (PI-87): Live inference.

Wraps the trained heads for use inside the real-time loop. Feature extraction
goes through the SAME `features` code as training, so online and offline never
drift.

Two ways to use it:

    # stateless — you already hold a window of frames
    pred = FormPredictor().predict_window(list_of_angle_dicts)

    # stateful — feed one frame at a time from the live loop (mirrors DataCollector)
    fp = FormPredictor()
    for angles in stream:
        result = fp.update(angles)      # None until the buffer fills, then a dict

A result dict looks like:
    {'head': 'quality', 'label': 'good', 'confidence': 0.82,
     'proba': {'good': 0.82, 'bad': 0.18}}

If the head has not been trained yet, the predictor loads as `available=False`
and every call returns None — so the pipeline can call it unconditionally and
simply ignore it until a model exists.
"""
import os
import sys
from collections import deque
from typing import Dict, Optional, Sequence

# Runnable as a plain script: put the package root (core/) on the path first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features import features_from_dicts, DEFAULT_WINDOW, FEATURE_NAMES
from ml import store


class FormPredictor:
    def __init__(self, head: str = 'quality', models_dir: str = None,
                 predict_every: int = 1):
        """
        head:          which trained model to serve ('quality' / 'phase' / 'exercise').
        predict_every: in stateful mode, only run the model every N frames (>=1) to
                       cut jitter and cost; the last result is returned in between.
        """
        self.head = head
        self.predict_every = max(1, predict_every)
        bundle = store.load_model(head)

        self.available = bundle is not None
        if self.available:
            self._pipeline = bundle['pipeline']
            self._classes = bundle['classes']
            self.kind = bundle.get('kind', 'classifier')
            self.window = bundle['window']
            # Guard against a model trained on a different feature layout.
            if bundle['feature_names'] != FEATURE_NAMES:
                raise ValueError(
                    f"'{head}' was trained on a different feature set; retrain it "
                    f"(run core/ml/trainer.py)."
                )
        else:
            self._pipeline = None
            self._classes = None
            self.kind = None
            self.window = DEFAULT_WINDOW

        self._buffer: deque = deque(maxlen=self.window)
        self._frame_counter = 0
        self._last: Optional[dict] = None

    # ── stateless ────────────────────────────────────────────────────────
    def predict_window(self, frames: Sequence[Dict[str, float]]) -> Optional[dict]:
        """Classify one full window of per-frame angle dicts. None if no model."""
        if not self.available or len(frames) == 0:
            return None
        x = features_from_dicts(frames).reshape(1, -1)
        if self.kind == 'oneclass':
            # IsolationForest: decision_function > 0 == inside the good-form envelope.
            score = float(self._pipeline.decision_function(x)[0])
            is_good = int(self._pipeline.predict(x)[0]) == 1
            return {'head': self.head, 'kind': 'oneclass',
                    'label': 'good' if is_good else 'off_form',
                    'score': round(score, 4), 'confidence': None, 'proba': None}
        return self._format(self._pipeline.predict(x)[0], self._proba(x))

    # ── stateful (per-frame) ─────────────────────────────────────────────
    def update(self, angles: Dict[str, float]) -> Optional[dict]:
        """Push one frame; return a prediction once the buffer is full, else None."""
        if not self.available:
            return None
        self._buffer.append(dict(angles))
        self._frame_counter += 1
        if len(self._buffer) < self.window:
            return None
        if self._frame_counter % self.predict_every != 0:
            return self._last
        self._last = self.predict_window(list(self._buffer))
        return self._last

    def reset(self):
        """Clear the rolling buffer (e.g. on a new exercise/session)."""
        self._buffer.clear()
        self._frame_counter = 0
        self._last = None

    # ── helpers ──────────────────────────────────────────────────────────
    def _proba(self, x) -> Optional[dict]:
        if not hasattr(self._pipeline, 'predict_proba'):
            return None
        probs = self._pipeline.predict_proba(x)[0]
        return {str(c): round(float(p), 4) for c, p in zip(self._classes, probs)}

    def _format(self, label, proba: Optional[dict]) -> dict:
        confidence = max(proba.values()) if proba else None
        return {'head': self.head, 'kind': 'classifier', 'label': str(label),
                'confidence': confidence, 'proba': proba}


if __name__ == '__main__':
    fp = FormPredictor('quality')
    if not fp.available:
        print("No 'quality' model yet. Train one with: python core/ml/trainer.py")
    else:
        print(f"Loaded 'quality' model (window={fp.window}, classes={fp._classes}).")
