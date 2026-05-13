import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from dataclasses import dataclass
from typing import Dict, Optional
import urllib.request
import os

@dataclass
class Point:
    x: float
    y: float
    z: float
    visibility: float

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".pose-iq")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker.task")

class PoseDetector:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            print("Downloading pose landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")

        original_dir = os.getcwd()
        os.chdir(MODEL_DIR)
        try:
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
        finally:
            os.chdir(original_dir)

    def find_pose(self, frame) -> Optional[Dict[int, Point]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        landmarks = {}
        h, w, c = frame.shape

        for id, lm in enumerate(results.pose_landmarks[0]):
            landmarks[id] = Point(x=lm.x * w, y=lm.y * h, z=lm.z * w, visibility=lm.visibility)

        return landmarks
