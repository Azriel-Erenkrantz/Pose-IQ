import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'user_profile.json')

FITNESS_LEVELS = {
    'beginner': {'label': 'Beginner', 'threshold_modifier': 1.3},
    'intermediate': {'label': 'Intermediate', 'threshold_modifier': 1.0},
    'advanced': {'label': 'Advanced', 'threshold_modifier': 0.85}
}

LIMITATION_OPTIONS = [
    'right_knee', 'left_knee',
    'lower_back',
    'right_shoulder', 'left_shoulder',
    'right_elbow', 'left_elbow'
]

JOINT_MAP = {
    'right_knee': ['right_knee'],
    'left_knee': ['left_knee'],
    'lower_back': ['spine'],
    'right_shoulder': ['right_arm_body'],
    'left_shoulder': ['left_arm_body'],
    'right_elbow': ['right_elbow'],
    'left_elbow': ['left_elbow']
}


@dataclass
class UserProfile:
    name: str = ""
    fitness_level: str = "intermediate"
    limitations: List[str] = field(default_factory=list)
    preferred_exercises: List[str] = field(default_factory=list)

    @property
    def threshold_modifier(self) -> float:
        return FITNESS_LEVELS.get(self.fitness_level, FITNESS_LEVELS['intermediate'])['threshold_modifier']

    @property
    def limited_joints(self) -> List[str]:
        joints = []
        for limitation in self.limitations:
            joints.extend(JOINT_MAP.get(limitation, []))
        return joints

    def save(self, path: str = None):
        path = path or PROFILE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str = None) -> Optional['UserProfile']:
        path = path or PROFILE_PATH
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return UserProfile(**data)

    @staticmethod
    def exists(path: str = None) -> bool:
        path = path or PROFILE_PATH
        return os.path.exists(path)
