"""
PI-90: One-off migration from the legacy JSON files into the database.

Reads data/user_profile.json + data/workout_history.json and writes them through
the ORM. Idempotent: an existing user is updated and existing sessions are skipped,
so re-running never duplicates.

Run:  python core/db/migrate_json.py
"""
import json
import os
import sys
import uuid
from datetime import datetime

# Runnable as a plain script: put the package root (core/) on the path first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import DATA_DIR, init_db, session_scope
from db.models import WorkoutSession
from db.repositories import UserRepository, WorkoutRepository

PROFILE_JSON = os.path.join(DATA_DIR, 'user_profile.json')
HISTORY_JSON = os.path.join(DATA_DIR, 'workout_history.json')


def _parse_date(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.now()


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def migrate():
    init_db()
    with session_scope() as s:
        users = UserRepository(s)
        workouts = WorkoutRepository(s)

        user_id = None
        if os.path.exists(PROFILE_JSON):
            p = _load_json(PROFILE_JSON)
            user_id = p.get('user_id') or uuid.uuid4().hex[:12]
            users.upsert(
                id=user_id,
                name=p.get('name', ''),
                fitness_level=p.get('fitness_level', 'intermediate'),
                coach_style=p.get('coach_style', 'motivator'),
                limitations=p.get('limitations', []),
                preferred_exercises=p.get('preferred_exercises', []),
            )
            print(f"  user {user_id} ({p.get('name', '?')}) migrated")
        else:
            print("  no user_profile.json — skipping user")

        n_migrated = n_skipped = 0
        if os.path.exists(HISTORY_JSON) and user_id:
            for sess in _load_json(HISTORY_JSON).get('sessions', []):
                if s.get(WorkoutSession, sess['session_id']) is not None:
                    n_skipped += 1
                    continue
                workouts.save_session(
                    user_id,
                    session_id=sess['session_id'],
                    exercise_id=sess['exercise_id'],
                    exercise_name=sess.get('exercise_name', ''),
                    date=_parse_date(sess.get('date')),
                    duration_seconds=sess.get('duration_seconds', 0.0),
                    total_reps=sess.get('total_reps', 0),
                    overall_score=sess.get('overall_score', 0.0),
                    weight_kg=sess.get('weight_kg'),     # absent in legacy data -> NULL
                    rep_records=sess.get('rep_records', []),
                )
                n_migrated += 1
            print(f"  sessions migrated: {n_migrated}, skipped (already present): {n_skipped}")
        elif not os.path.exists(HISTORY_JSON):
            print("  no workout_history.json — no sessions to migrate")

    print("Migration complete.")


if __name__ == '__main__':
    migrate()
