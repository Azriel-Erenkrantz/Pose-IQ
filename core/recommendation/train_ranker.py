"""
Force a retrain of the recommendation ranker right now, against whatever
ratings currently exist in Mongo.

api/app.py already does this automatically once per server process (on
the first request) — this CLI is for forcing a fresh retrain without
restarting the server, or for local inspection.

Run:
    python -m core.recommendation.train_ranker
"""
from __future__ import annotations

from .ranker import MODEL_PATH, train


def main() -> None:
    train()
    print(f'Trained ranker written to {MODEL_PATH}')


if __name__ == '__main__':
    main()
