"""
PI-90: Database connection — engine + session factory.

DB-agnostic on purpose: develop on SQLite, deploy on PostgreSQL by changing only
the DATABASE_URL env var. The ORM models and repositories are identical for both,
so the RN-vs-web client decision never touches this layer.

    dev  (default):  sqlite:///<project>/data/poseiq.db
    prod:            postgresql+psycopg://user:pass@host:5432/poseiq
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DEFAULT_SQLITE = 'sqlite:///' + os.path.join(DATA_DIR, 'poseiq.db')
DATABASE_URL = os.environ.get('DATABASE_URL', DEFAULT_SQLITE)


class Base(DeclarativeBase):
    pass


# check_same_thread only applies to SQLite (lets the connection cross threads,
# which a web server needs); ignored for PostgreSQL.
_connect_args = {'check_same_thread': False} if DATABASE_URL.startswith('sqlite') else {}
engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db():
    """Create all tables if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    import db.models  # noqa: F401 — register models on Base before create_all
    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    """Transactional scope: commit on success, roll back on error, always close."""
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
