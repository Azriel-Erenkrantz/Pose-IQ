"""
Train the rating-based ranker on fake data and write the result to
data/recommendation_ranker.json (committed — not the gitignored
data/models/, so it ships with the deployed API).

Run after changing fake_ratings.py:
    python -m core.recommendation.train_ranker
"""
from __future__ import annotations

from .ranker import MODEL_PATH, train


def main() -> None:
    train()
    print(f'Trained ranker written to {MODEL_PATH}')


if __name__ == '__main__':
    main()
