"""
Step 4 (feeds PI-84/85): Offline extractor.

Turns labelled training videos into the SAME CSV schema the live collector
writes, so both feed one training pipeline.

Expected layout (the label is encoded in the path):
    data/videos/<exercise_id>/good/clip.mp4     -> quality_label = 'good'
    data/videos/<exercise_id>/bad/clip.mp4      -> quality_label = 'bad'
    data/videos/<exercise_id>/clip.mp4          -> quality_label = 'unknown'

  exercise label  = top folder (must match an id in exercises.json)
  quality label   = 'good' / 'bad' subfolder, else 'unknown'
  phase per frame = ExerciseModel.match_phase (stateless best-match, no FSM)
  quality_score   = rule-based weak label (100 - 15 * violations), matching live

Run:  python core/ml/video_extractor.py
"""
import os
import sys

# Make this runnable as a plain script: put the package root (core/) on the path
# before the project imports below resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import uuid

import cv2

from ml.data_collector import HEADER, build_row, SAMPLE_EVERY_N_FRAMES
from detection.pose_detector import PoseDetector
from detection.angle_calculator import AngleCalculator
from exercise.exercise_model import ExerciseModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
VIDEO_CSV = os.path.join(DATA_DIR, 'video_data.csv')

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
QUALITY_DIRS = {'good', 'bad'}


def iter_videos(root):
    """Yield (path, exercise_id, quality_label) for every video file under root."""
    for exercise_id in sorted(os.listdir(root)):
        ex_dir = os.path.join(root, exercise_id)
        if not os.path.isdir(ex_dir):
            continue
        for dirpath, _dirs, files in os.walk(ex_dir):
            parts = os.path.relpath(dirpath, ex_dir).split(os.sep)
            quality = parts[0] if parts and parts[0] in QUALITY_DIRS else 'unknown'
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                    yield os.path.join(dirpath, fn), exercise_id, quality


def process_video(path, exercise_id, quality_label, detector, model, writer):
    """Run pose detection over one video and write sampled rows. Returns row count."""
    session_id = 'vid_' + uuid.uuid4().hex[:8]      # one session per clip -> windows never cross clips
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  ! could not open {path}")
        return 0

    frame_idx = 0
    rows = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:        # match the live sampling rate
            continue

        landmarks = detector.find_pose(frame)
        if not landmarks:
            continue
        angles = AngleCalculator.get_body_angles(landmarks)
        if not angles:
            continue

        match = model.match_phase(exercise_id, angles)
        if match:
            phase, violations = match
            phase_name, phase_index, n_viol = phase.name, phase.order, len(violations)
        else:
            phase_name, phase_index, n_viol = 'unknown', -1, 0

        writer.writerow(build_row(
            user_id='video',
            session_id=session_id,
            exercise_id=exercise_id,
            rep_number=-1,                  # reps are not tracked offline
            phase_name=phase_name,
            phase_index=phase_index,
            angles=angles,
            in_phase=(n_viol == 0),
            quality_score=max(0, 100 - 15 * n_viol),
            quality_label=quality_label,
        ))
        rows += 1

    cap.release()
    return rows


def main(video_dir=VIDEO_DIR, out_csv=VIDEO_CSV):
    if not os.path.isdir(video_dir):
        print(f"No video folder at {video_dir}.")
        print("Create it as:  data/videos/<exercise_id>/<good|bad>/clip.mp4")
        return

    model = ExerciseModel()
    known = set(model.list_exercises())
    detector = PoseDetector()

    new_file = not os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    total, clips = 0, 0
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        for path, exercise_id, quality in iter_videos(video_dir):
            if exercise_id not in known:
                print(f"  - skip {os.path.basename(path)} (unknown exercise '{exercise_id}'; known: {sorted(known)})")
                continue
            n = process_video(path, exercise_id, quality, detector, model, writer)
            total += n
            clips += 1
            print(f"  + {os.path.basename(path)} [{exercise_id}/{quality}] -> {n} rows")

    print(f"\nDone. {clips} clips, {total} rows -> {out_csv}")


if __name__ == '__main__':
    main()
