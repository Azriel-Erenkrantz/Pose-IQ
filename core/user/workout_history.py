import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data', 'workout_history.json'
)

JOINT_TO_MUSCLE = {
    'right_knee': 'Quadriceps / Right Knee',
    'left_knee': 'Quadriceps / Left Knee',
    'spine': 'Core / Lower Back',
    'right_elbow': 'Triceps / Right Elbow',
    'left_elbow': 'Triceps / Left Elbow',
    'right_arm_body': 'Shoulders / Right',
    'left_arm_body': 'Shoulders / Left',
    'legs_spread': 'Hip Stability',
}


@dataclass
class RepRecord:
    rep_number: int
    error_joints: List[str] = field(default_factory=list)
    form_score: float = 100.0


@dataclass
class WorkoutSession:
    session_id: str
    date: str
    exercise_id: str
    exercise_name: str
    total_reps: int
    rep_records: List[RepRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    overall_score: float = 0.0

    def calculate_score(self):
        if not self.rep_records:
            self.overall_score = 0.0
            return
        self.overall_score = round(
            sum(r.form_score for r in self.rep_records) / len(self.rep_records), 1
        )


class WorkoutHistory:
    def __init__(self, path: str = None):
        self.path = path or HISTORY_PATH
        self.sessions: List[WorkoutSession] = self._load()

    def _load(self) -> List[WorkoutSession]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sessions = []
        for s in data.get('sessions', []):
            rep_records = [RepRecord(**r) for r in s.get('rep_records', [])]
            s_data = {k: v for k, v in s.items() if k != 'rep_records'}
            sessions.append(WorkoutSession(**s_data, rep_records=rep_records))
        return sessions

    def save_session(self, session: WorkoutSession):
        session.calculate_score()
        self.sessions.append(session)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump({'sessions': [asdict(s) for s in self.sessions]}, f, indent=2, ensure_ascii=False)

    def get_sessions(self, exercise_id: str = None, limit: int = None) -> List[WorkoutSession]:
        result = self.sessions
        if exercise_id:
            result = [s for s in result if s.exercise_id == exercise_id]
        result = sorted(result, key=lambda s: s.date, reverse=True)
        return result[:limit] if limit else result

    def get_progress_metrics(self, exercise_id: str) -> dict:
        sessions = self.get_sessions(exercise_id)
        if not sessions:
            return {}

        recent = sessions[:5]
        scores = [s.overall_score for s in recent]
        reps = [s.total_reps for s in recent]

        joint_errors: Dict[str, int] = {}
        total_reps = 0
        for s in sessions:
            for rep in s.rep_records:
                total_reps += 1
                for joint in rep.error_joints:
                    joint_errors[joint] = joint_errors.get(joint, 0) + 1

        weak_joints = sorted(joint_errors.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_sessions': len(sessions),
            'total_reps': sum(s.total_reps for s in sessions),
            'avg_score_recent': round(sum(scores) / len(scores), 1) if scores else 0.0,
            'avg_reps_recent': round(sum(reps) / len(reps), 1) if reps else 0.0,
            'score_trend': _trend(scores),
            'weak_joints': weak_joints[:3],
        }

    def get_weak_muscle_groups(self) -> List[Tuple[str, int]]:
        joint_errors: Dict[str, int] = {}
        for s in self.sessions:
            for rep in s.rep_records:
                for joint in rep.error_joints:
                    joint_errors[joint] = joint_errors.get(joint, 0) + 1
        sorted_joints = sorted(joint_errors.items(), key=lambda x: x[1], reverse=True)
        return sorted_joints[:3]


def _trend(values: List[float]) -> str:
    if len(values) < 2:
        return 'neutral'
    mid = len(values) // 2
    avg_old = sum(values[mid:]) / max(len(values) - mid, 1)
    avg_new = sum(values[:mid]) / max(mid, 1)
    if avg_new > avg_old + 2:
        return 'improving'
    if avg_new < avg_old - 2:
        return 'declining'
    return 'stable'


def new_session(exercise_id: str, exercise_name: str) -> WorkoutSession:
    return WorkoutSession(
        session_id=str(uuid.uuid4())[:8],
        date=datetime.now().isoformat(),
        exercise_id=exercise_id,
        exercise_name=exercise_name,
        total_reps=0
    )


def rep_form_score(error_joints: List[str]) -> float:
    return max(0.0, round(100.0 - len(set(error_joints)) * 15.0, 1))
