import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.db import get_db

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
    weight_kg: Optional[float] = None   # load used; None = not logged, 0 = bodyweight

    def calculate_score(self):
        if not self.rep_records:
            self.overall_score = 0.0
            return
        self.overall_score = round(
            sum(r.form_score for r in self.rep_records) / len(self.rep_records), 1
        )


class WorkoutHistory:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def _doc_to_session(self, doc: dict) -> WorkoutSession:
        reps = [RepRecord(**r) for r in doc.get('rep_records', [])]
        date = doc['date']
        if isinstance(date, datetime):
            date = date.isoformat()
        return WorkoutSession(
            session_id       = str(doc['_id']),
            date             = date,
            exercise_id      = doc['exercise_id'],
            exercise_name    = doc['exercise_name'],
            total_reps       = doc['total_reps'],
            rep_records      = reps,
            duration_seconds = doc.get('duration_seconds', 0.0),
            overall_score    = doc.get('overall_score', 0.0),
            weight_kg        = doc.get('weight_kg'),
        )

    def save_session(self, session: WorkoutSession):
        session.calculate_score()
        get_db().sessions.insert_one({
            '_id':             session.session_id,
            'user_id':         self.user_id,
            'date':            session.date,
            'exercise_id':     session.exercise_id,
            'exercise_name':   session.exercise_name,
            'total_reps':      session.total_reps,
            'rep_records':     [{'rep_number': r.rep_number, 'error_joints': r.error_joints, 'form_score': r.form_score} for r in session.rep_records],
            'duration_seconds': session.duration_seconds,
            'overall_score':   session.overall_score,
            'weight_kg':       session.weight_kg,
        })

    def set_session_weight(self, session_id: str, weight_kg: Optional[float]) -> bool:
        """Set the load used in a past session. Scoped to this user — a
        session id belonging to someone else is not found. Returns False
        when no matching session exists."""
        result = get_db().sessions.update_one(
            {'_id': session_id, 'user_id': self.user_id},
            {'$set': {'weight_kg': weight_kg}},
        )
        return result.matched_count > 0


    def get_sessions(self, exercise_id: str = None, limit: int = None) -> List[WorkoutSession]:
        query: dict = {'user_id': self.user_id}
        if exercise_id:
            query['exercise_id'] = exercise_id
        cursor = get_db().sessions.find(query).sort('date', -1)
        if limit:
            cursor = cursor.limit(limit)
        return [self._doc_to_session(doc) for doc in cursor]

    @property
    def sessions(self) -> List[WorkoutSession]:
        return self.get_sessions()

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


def new_session(exercise_id: str, exercise_name: str,
                weight_kg: Optional[float] = None) -> WorkoutSession:
    return WorkoutSession(
        session_id=str(uuid.uuid4())[:8],
        date=datetime.now().isoformat(),
        exercise_id=exercise_id,
        exercise_name=exercise_name,
        total_reps=0,
        weight_kg=weight_kg,
    )


def rep_form_score(error_joints: List[str]) -> float:
    return max(0.0, round(100.0 - len(set(error_joints)) * 15.0, 1))
