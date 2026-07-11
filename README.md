# Pose-IQ

**AI personal workout coach** — real-time form correction from a live camera feed,
per-user workout history, and personalized exercise recommendations.

The camera captures the athlete, MediaPipe detects a 33-point body skeleton, joint
angles are computed per frame, a state machine tracks the exercise phase and counts
reps, a rules engine flags posture mistakes with visual + voice feedback, and every
session feeds a recommendation engine that learns the user over time.

Exercises: **squat, lunge, biceps curl, shoulder press**.

## Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│ Live pipeline (desktop)      │      │ Server                       │
│ camera → MediaPipe skeleton  │      │ Flask REST API (api/app.py)  │
│ → joint angles → state       │─────▶│ auth · profile · history ·   │
│ machine (phases, reps)       │ save │ progress · recommendations   │
│ → posture rules → HUD + TTS  │      │        │                     │
│ + RF phase classifier (ML)   │      │     MongoDB                  │
└─────────────────────────────┘      └──────────┬───────────────────┘
                                                │ REST
                                     ┌──────────┴───────────────────┐
                                     │ Clients                      │
                                     │ frontend/ — React web app    │
                                     │ mobile/   — Expo RN scaffold │
                                     └──────────────────────────────┘
```

| Layer | Path | Role |
|-------|------|------|
| Detection | `core/detection/` | Camera stream, MediaPipe pose, 3D angle calculator |
| Exercise logic | `core/exercise/` | Exercise model (Mongo/seed JSON), phase state machine, posture rules |
| ML | `core/ml/` | Video → landmarks → angles, hand-labeled phase training (Random Forest per exercise), Gaussian fallback classifier, violation detector, evaluation |
| Users | `core/user/` | Service layer (auth, profile, health, history, progress) over MongoDB |
| Recommendations | `core/recommendation/` | Health-scenario scoring, community/feedback blending, ML ranker with rule fallback |
| Coaching | `core/coaching/` | TTS voice coach, 3 personality styles |
| Contracts | `core/app_model.py` | Single source of truth for all data types (API ⇄ clients) |
| API | `api/app.py` | Flask REST API |
| Web | `frontend/` | React + Vite (login, onboarding, dashboard) |
| Mobile | `mobile/` | Expo React Native scaffold |

## Setup

Requirements: Python 3.12, MongoDB running locally (or `MONGODB_URI` env var), Node 18+ for the web app.

```bash
pip install -r requirements.txt

# one-time DB setup
python -m core.exercise.seed        # seed exercise definitions into MongoDB
python -m core.ml.trainer           # train phase models from labeled videos
                                    # (also writes measured angle ranges to Mongo)
```

Training videos are not in git (~190MB). Label files live in `data/labeled_vidz/{exercise}/*.json`
and reference videos under `data/videos/{exercise}/good/`.

## Running

**Live workout (desktop, needs a webcam):**

```bash
python -m core.pipeline squat <user_id>     # user_id comes from registration
```

Keys: `q` quit · `d` toggle debug overlay (per-joint angles + ML phase classifier vs state machine).

**API server:**

```bash
flask --app api.app run --debug             # http://localhost:5000
```

**Web app:**

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

**Tests:**

```bash
python -m pytest tests/ -q
```

## ML training & evaluation

- `python -m core.ml.trainer` — trains a Random Forest phase classifier per exercise
  from hand-labeled videos; holds out the last label file per exercise for evaluation.
- `python -m core.ml.eval "data/labeled_vidz/squat/<file>.json"` — accuracy, per-phase
  breakdown and confusion matrix on a labeled video (`--gaussian` compares the
  non-ML fallback).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | `poseiq` | Database name |
