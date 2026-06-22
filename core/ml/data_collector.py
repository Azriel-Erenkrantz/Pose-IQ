"""
PI-46: Data Collection Pipeline
Collects angle snapshots per frame with automatic tagging.
Each row = one sampled frame during an active exercise session.
"""
import csv
import os
import uuid
from typing import Dict, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
TRAINING_CSV = os.path.join(DATA_DIR, 'training_data.csv')

ANGLE_COLS = [
    'right_knee', 'left_knee', 'spine', 'legs_spread',
    'right_elbow', 'left_elbow', 'right_arm_body', 'left_arm_body',
]

HEADER = (
    ['user_id', 'session_id', 'exercise_id', 'rep_number', 'phase_name', 'phase_index']
    + ANGLE_COLS
    + ['in_phase', 'quality_score', 'quality_label']
)

SAMPLE_EVERY_N_FRAMES = 3
MIN_ROWS_TO_RETRAIN   = 500


def build_row(
    user_id: str,
    session_id: str,
    exercise_id: str,
    rep_number: int,
    phase_name: str,
    phase_index: int,
    angles: Dict[str, float],
    in_phase: bool,
    quality_score: float,
    quality_label: str = 'live',
) -> list:
    """
    Single source of truth for a CSV row, shared by the live collector and the
    offline video extractor so both always match HEADER.

    quality_label: 'live' for real sessions (good/bad unknown), or 'good'/'bad'
    for clip-level labelled training videos.
    """
    return (
        [user_id, session_id, exercise_id, rep_number, phase_name, phase_index]
        + [round(angles.get(col, -1.0), 1) for col in ANGLE_COLS]
        + [int(in_phase), round(quality_score, 1), quality_label]
    )


class DataCollector:
    def __init__(self, path: str = None, user_id: str = "anonymous"):
        self.path = path or TRAINING_CSV
        self._user_id = user_id
        self._session_id = str(uuid.uuid4())[:8]
        self._frame_counter = 0
        self._buffer: list = []
        self._buffer_size = 30          # flush every 30 collected rows
        self._ensure_csv()

    def _ensure_csv(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(HEADER)

    def collect(
        self,
        exercise_id: str,
        rep_number: int,
        phase_name: str,
        phase_index: int,
        angles: Dict[str, float],
        in_phase: bool,
        quality_score: float,
    ):
        self._frame_counter += 1
        if self._frame_counter % SAMPLE_EVERY_N_FRAMES != 0:
            return

        row = build_row(
            self._user_id, self._session_id, exercise_id, rep_number,
            phase_name, phase_index, angles, in_phase, quality_score,
            quality_label='live',
        )
        self._buffer.append(row)

        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(self._buffer)
        self._buffer.clear()

    def finish(self):
        self._flush()

    def row_count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, encoding='utf-8') as f:
            return sum(1 for _ in f) - 1          # subtract header

    def needs_retrain(self) -> bool:
        return self.row_count() >= MIN_ROWS_TO_RETRAIN


def export_to_parquet(csv_path: str = None, out_path: str = None):
    """Export training CSV to Parquet for faster ML loading."""
    import pandas as pd
    csv_path = csv_path or TRAINING_CSV
    out_path = out_path or csv_path.replace('.csv', '.parquet')
    df = pd.read_csv(csv_path)
    df.to_parquet(out_path, index=False)
    return out_path, len(df)
