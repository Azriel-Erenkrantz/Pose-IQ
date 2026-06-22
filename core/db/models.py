"""
PI-90: ORM models — a faithful relational mapping of the legacy JSON storage
(user_profile.json + workout_history.json), plus `weight_kg` on sessions, which
is NEW and required before any load/progression recommendation can exist.

    users ──< workout_sessions ──< rep_records ──< rep_errors

rep_errors is normalised (one row per joint error) on purpose: the "weak muscle
groups" analytics are a GROUP BY joint COUNT(*) — trivial and fast in SQL.
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(32), primary_key=True)   # stable user_id
    name: Mapped[str] = mapped_column(String(120), default='')
    fitness_level: Mapped[str] = mapped_column(String(20), default='intermediate')
    coach_style: Mapped[str] = mapped_column(String(20), default='motivator')
    # Small lists the app reads as-is; JSON keeps them faithful to the dataclass.
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    preferred_exercises: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    sessions: Mapped[list['WorkoutSession']] = relationship(
        back_populates='user', cascade='all, delete-orphan')


class WorkoutSession(Base):
    __tablename__ = 'workout_sessions'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)   # session_id
    user_id: Mapped[str] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'), index=True)
    exercise_id: Mapped[str] = mapped_column(String(40), index=True)
    exercise_name: Mapped[str] = mapped_column(String(80), default='')
    date: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    total_reps: Mapped[int] = mapped_column(Integer, default=0)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)   # NEW: load used

    user: Mapped['User'] = relationship(back_populates='sessions')
    reps: Mapped[list['RepRecord']] = relationship(
        back_populates='session', cascade='all, delete-orphan')


class RepRecord(Base):
    __tablename__ = 'rep_records'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey('workout_sessions.id', ondelete='CASCADE'), index=True)
    rep_number: Mapped[int] = mapped_column(Integer)
    form_score: Mapped[float] = mapped_column(Float, default=100.0)

    session: Mapped['WorkoutSession'] = relationship(back_populates='reps')
    errors: Mapped[list['RepError']] = relationship(
        back_populates='rep', cascade='all, delete-orphan')


class RepError(Base):
    __tablename__ = 'rep_errors'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rep_id: Mapped[int] = mapped_column(
        ForeignKey('rep_records.id', ondelete='CASCADE'), index=True)
    joint: Mapped[str] = mapped_column(String(40), index=True)

    rep: Mapped['RepRecord'] = relationship(back_populates='errors')
