"""
Video → per-frame pose landmarks using MediaPipe Tasks API (≥0.10).

The model file is auto-downloaded to data/models/ on first use.
Each frame returns normalized (x, y, z) joint positions — MediaPipe normalizes
by the person's own body size, so height/weight are not needed.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
)
# Cached under the user's home dir (like detection/pose_detector.py) — MediaPipe's
# C++ layer fails on non-ASCII paths, and the project may live in one.
DEFAULT_MODEL_PATH = Path.home() / '.pose-iq' / 'pose_landmarker_full.task'

# MediaPipe landmark index → readable name
LANDMARK_NAMES: Dict[int, str] = {
    0:  'nose',
    11: 'left_shoulder',  12: 'right_shoulder',
    13: 'left_elbow',     14: 'right_elbow',
    15: 'left_wrist',     16: 'right_wrist',
    23: 'left_hip',       24: 'right_hip',
    25: 'left_knee',      26: 'right_knee',
    27: 'left_ankle',     28: 'right_ankle',
    29: 'left_heel',      30: 'right_heel',
    31: 'left_foot_index',32: 'right_foot_index',
}

Point3D = Tuple[float, float, float]


@dataclass
class PoseFrame:
    frame_idx:    int
    timestamp_ms: float
    landmarks:    Dict[str, Point3D]   # joint name → (x, y, z) normalized 0-1
    visibility:   Dict[str, float]     # joint name → 0-1 confidence


def _ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the pose landmarker model if not already present."""
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading pose model to {model_path} ...')
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print('Download complete.')
    return model_path


def extract_frames(video_path: Path, skip: int = 1, model_path: Path | None = None) -> List[PoseFrame]:
    """
    Run MediaPipe Pose on a video file.

    Args:
        video_path:  path to .mp4 / .mov / etc.
        skip:        process every (skip)th frame — 1 = every frame.
        model_path:  path to .task model file (auto-downloaded if None).

    Returns list of PoseFrame, one per processed frame where a person was detected.
    """
    import mediapipe as mp

    resolved_model = _ensure_model(model_path or DEFAULT_MODEL_PATH)

    BaseOptions         = mp.tasks.BaseOptions
    PoseLandmarker      = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode         = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resolved_model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames: List[PoseFrame] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # MediaPipe checks the cwd at creation time and aborts on non-ASCII paths
    # (e.g. a project folder with Hebrew in it) — create from an ASCII-safe dir.
    import os
    original_dir = os.getcwd()
    os.chdir(resolved_model.parent)
    try:
        landmarker_ctx = PoseLandmarker.create_from_options(options)
    finally:
        os.chdir(original_dir)

    with landmarker_ctx as landmarker:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % skip == 0:
                timestamp_ms = int(idx / fps * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    raw = result.pose_landmarks[0]   # first (only) person
                    landmarks:  Dict[str, Point3D] = {}
                    visibility: Dict[str, float]   = {}

                    for lm_idx, name in LANDMARK_NAMES.items():
                        if lm_idx < len(raw):
                            lm = raw[lm_idx]
                            landmarks[name]  = (lm.x, lm.y, lm.z)
                            visibility[name] = getattr(lm, 'visibility', 1.0)

                    frames.append(PoseFrame(
                        frame_idx    = idx,
                        timestamp_ms = timestamp_ms,
                        landmarks    = landmarks,
                        visibility   = visibility,
                    ))
            idx += 1

    cap.release()
    return frames
