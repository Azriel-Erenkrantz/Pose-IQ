"""
Export trained phase classifiers (data/models/phase_{ex}.joblib) to ONNX,
for onnxruntime-web to run client-side in the browser.

Input: a single float32 tensor, shape [N, 24] — 12 joint angles followed by
12 per-joint angular velocities, in JOINTS order (core.ml.trainer.JOINTS).
Same layout as core.ml.trainer.angles_to_features(); the frontend must build
the vector in that exact order.

Output: two tensors — the predicted phase label (string) and per-class
probabilities (used for a confidence-gated fallback to the rule-based state
machine on the frontend, see roadmap #10 in CLAUDE.md).

Usage:
    python -m core.ml.export_onnx                 # all 4 exercises
    python -m core.ml.export_onnx --exercise squat
    python -m core.ml.export_onnx --skip-verify    # skip the prediction-parity check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np

from core.ml.trainer import JOINTS, MODELS_DIR

N_FEATURES = 2 * len(JOINTS)
OUT_DIR = Path(__file__).parent.parent.parent / 'frontend' / 'public' / 'models'
EXERCISE_IDS = ['squat', 'lunge', 'biceps_curl', 'shoulder_press']


def export_one(exercise_id: str, verify: bool = True) -> Path | None:
    src = MODELS_DIR / f'phase_{exercise_id}.joblib'
    if not src.exists():
        print(f'  [{exercise_id}] no model at {src} — skipping')
        return None

    bundle = joblib.load(src)
    clf = bundle['model']

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(
        clf,
        initial_types=[('features', FloatTensorType([None, N_FEATURES]))],
        target_opset=17,
        # Default sklearn->ONNX classifier export wraps predict_proba in a
        # ZipMap (sequence<map<string,float>>) — onnxruntime-web's WASM
        # backend can't marshal that back to JS at all (confirmed 1.29.0;
        # session.run() throws for the whole call if it's fetched). A plain
        # [N, n_classes] float tensor works fine; the frontend maps columns
        # back to phase names via `classes` in the same order (see
        # phaseClassifier.ts).
        options={id(clf): {'zipmap': False}},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'phase_{exercise_id}.onnx'
    out_path.write_bytes(onnx_model.SerializeToString())
    size_kb = out_path.stat().st_size / 1024
    print(f'  [{exercise_id}] {src.name} ({src.stat().st_size / 1024:.0f}KB) '
          f'→ {out_path.relative_to(OUT_DIR.parent.parent.parent)} ({size_kb:.0f}KB)')

    if verify:
        _verify_parity(exercise_id, clf, out_path)

    return out_path


def _verify_parity(exercise_id: str, clf, onnx_path: Path, n_samples: int = 500) -> None:
    """The whole point of ONNX export is that it behaves identically to the
    sklearn model it came from — never ship a conversion without checking
    that on real-shaped random input, not just 'it exported without error'."""
    import onnxruntime as ort

    rng = np.random.default_rng(42)
    # Angles roughly in plausible joint-angle range; deltas centered on 0 —
    # doesn't need to be realistic, just needs to exercise the same decision
    # boundaries sklearn and the ONNX graph were both built from.
    angles = rng.uniform(0, 180, size=(n_samples, len(JOINTS))).astype(np.float32)
    deltas = rng.uniform(-60, 60, size=(n_samples, len(JOINTS))).astype(np.float32)
    X = np.concatenate([angles, deltas], axis=1)

    sk_pred = clf.predict(X)

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {input_name: X})[0]

    agree = np.array([str(a) == str(b) for a, b in zip(sk_pred, onnx_pred)])
    rate = agree.mean()
    status = 'OK' if rate == 1.0 else 'MISMATCH'
    print(f'  [{exercise_id}] parity check: {agree.sum()}/{n_samples} agree ({rate:.1%}) — {status}')
    if rate < 1.0:
        raise RuntimeError(
            f'{exercise_id}: ONNX export disagrees with sklearn on '
            f'{n_samples - agree.sum()}/{n_samples} random samples — do not ship this .onnx file.'
        )


def main() -> None:
    sys.stdout.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exercise', type=str, default=None, help='Export only this exercise')
    parser.add_argument('--skip-verify', action='store_true', help='Skip the prediction-parity check')
    args = parser.parse_args()

    ids: List[str] = [args.exercise] if args.exercise else EXERCISE_IDS
    print(f'Exporting to {OUT_DIR}/')
    for ex_id in ids:
        export_one(ex_id, verify=not args.skip_verify)


if __name__ == '__main__':
    main()
