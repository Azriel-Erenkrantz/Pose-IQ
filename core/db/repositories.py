"""
PI-90: Repositories — thin data-access over the ORM, mirroring the queries the app
already does (workout_history.py / workout_plan.py) but DB-backed and per-user.

The point of moving to SQL shows up in `weak_muscle_groups` / `progress_metrics`:
what is a hand-rolled Python loop today becomes a single GROUP BY aggregation.
"""
from typing import List, Optional, Tuple

from sqlalchemy import func, select

from db.models import RepError, RepRecord, User, WorkoutSession


class UserRepository:
    def __init__(self, session):
        self.s = session

    def get(self, user_id: str) -> Optional[User]:
        return self.s.get(User, user_id)

    def upsert(self, *, id: str, name: str = '', fitness_level: str = 'intermediate',
               coach_style: str = 'motivator', limitations=None,
               preferred_exercises=None) -> User:
        u = self.s.get(User, id)
        if u is None:
            u = User(id=id)
            self.s.add(u)
        u.name = name
        u.fitness_level = fitness_level
        u.coach_style = coach_style
        u.limitations = list(limitations or [])
        u.preferred_exercises = list(preferred_exercises or [])
        return u


class WorkoutRepository:
    def __init__(self, session):
        self.s = session

    def save_session(self, user_id: str, *, session_id: str, exercise_id: str,
                     exercise_name: str, date, duration_seconds: float, total_reps: int,
                     overall_score: float, weight_kg: float = None,
                     rep_records=()) -> WorkoutSession:
        ws = WorkoutSession(
            id=session_id, user_id=user_id, exercise_id=exercise_id,
            exercise_name=exercise_name, date=date, duration_seconds=duration_seconds,
            total_reps=total_reps, overall_score=overall_score, weight_kg=weight_kg,
        )
        for r in rep_records:
            rep = RepRecord(rep_number=r['rep_number'], form_score=r.get('form_score', 100.0))
            rep.errors = [RepError(joint=j) for j in r.get('error_joints', [])]
            ws.reps.append(rep)
        self.s.add(ws)
        return ws

    def get_sessions(self, user_id: str, exercise_id: str = None,
                     limit: int = None) -> List[WorkoutSession]:
        q = select(WorkoutSession).where(WorkoutSession.user_id == user_id)
        if exercise_id:
            q = q.where(WorkoutSession.exercise_id == exercise_id)
        q = q.order_by(WorkoutSession.date.desc())
        if limit:
            q = q.limit(limit)
        return list(self.s.scalars(q))

    def weak_muscle_groups(self, user_id: str, top: int = 3) -> List[Tuple[str, int]]:
        """(joint, error_count) for this user's most-violated joints — one GROUP BY."""
        q = (
            select(RepError.joint, func.count().label('n'))
            .join(RepRecord, RepError.rep_id == RepRecord.id)
            .join(WorkoutSession, RepRecord.session_id == WorkoutSession.id)
            .where(WorkoutSession.user_id == user_id)
            .group_by(RepError.joint)
            .order_by(func.count().desc())
            .limit(top)
        )
        return [(joint, n) for joint, n in self.s.execute(q)]

    def recent_scores(self, user_id: str, exercise_id: str, limit: int = 5) -> List[float]:
        q = (
            select(WorkoutSession.overall_score)
            .where(WorkoutSession.user_id == user_id,
                   WorkoutSession.exercise_id == exercise_id)
            .order_by(WorkoutSession.date.desc())
            .limit(limit)
        )
        return list(self.s.scalars(q))
