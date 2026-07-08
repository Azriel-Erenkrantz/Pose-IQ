"""
MongoDB connection singleton.

Reads MONGODB_URI and MONGODB_DB from environment (or .env via python-dotenv).
All other modules call get_db() — never import MongoClient directly.

Swap point for tests: patch core.user.service.get_db / core.user.workout_history.get_db
with a mongomock database to run without a real MongoDB instance.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

load_dotenv()

_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        uri  = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        name = os.getenv('MONGODB_DB', 'poseiq')
        _db  = MongoClient(uri, serverSelectionTimeoutMS=5000)[name]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db: Database) -> None:
    db.users.create_index('email', unique=True)
    db.tokens.create_index('expires_at', expireAfterSeconds=0)   # TTL — auto-delete expired tokens
    db.sessions.create_index([('user_id', ASCENDING), ('date', ASCENDING)])
    db.exercises.create_index('id', unique=True)
    db.exercise_angles.create_index(
        [('exercise_id', ASCENDING), ('phase', ASCENDING)], unique=True
    )
