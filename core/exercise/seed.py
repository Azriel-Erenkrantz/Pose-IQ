"""
Seed the MongoDB exercises collection from data/exercises_seed.json.

Run once after schema changes or on a fresh database:
    python -m core.exercise.seed
    python -m core.exercise.seed --clear    # drops existing documents first

Idempotent: uses replace_one+upsert so running twice is safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED_PATH = Path(__file__).parent.parent.parent / 'data' / 'exercises_seed.json'


def seed(clear: bool = False, path: Path = SEED_PATH) -> None:
    from core.db import get_db
    db = get_db()

    data = json.loads(path.read_text(encoding='utf-8'))
    exercises = data['exercises']

    if clear:
        result = db.exercises.delete_many({})
        print(f'Cleared {result.deleted_count} documents from exercises collection.')

    for ex in exercises:
        db.exercises.replace_one({'id': ex['id']}, ex, upsert=True)
        print(f'  upserted: {ex["id"]}')

    print(f'Done. {len(exercises)} exercises seeded.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Seed MongoDB exercises collection from exercises_seed.json')
    parser.add_argument('--clear', action='store_true', help='Delete all existing exercises before inserting')
    parser.add_argument('--path', type=Path, default=SEED_PATH, help='Path to seed JSON (default: data/exercises_seed.json)')
    args = parser.parse_args()
    seed(clear=args.clear, path=args.path)


if __name__ == '__main__':
    main()
