# Pose-IQ — Claude Code Project Guide

Final-year project (פרויקט גמר): AI personal workout coach — real-time form
correction from a live camera, per-user history, personalized recommendations.
Goal: cloud deployment with real users. The user communicates in Hebrew;
code, comments, and commits are in English.

## Stack & architecture

- **Live pipeline** (`core/pipeline.py`): OpenCV window — camera → MediaPipe
  (33 landmarks) → joint angles → phase state machine (reps) → posture rules →
  HUD + TTS coach. Run: `python -m core.pipeline squat <user_id>`. Key `d`
  toggles the debug overlay (angles + ML phase vs state machine).
- **ML** (`core/ml/`): hand-labeled videos (`data/labeled_vidz/{ex}/*.json`) →
  per-exercise RandomForest phase classifier (`data/models/phase_{ex}.joblib`).
  `trainer.py` also writes measured per-phase angle stats to Mongo
  `exercise_angles` — those ranges drive the state machine.
- **DB**: MongoDB (`core/db/get_db()`, `MONGODB_URI` env, default localhost).
  Collections: users, tokens (TTL), sessions, exercises, exercise_angles.
  The old SQLAlchemy/Postgres layer was deleted — docs mentioning it are stale.
- **API** (`api/app.py`, Flask): auth, profile, health ratings, history,
  progress, dashboard (embeds recommendations). Run: `flask --app api.app run`.
- **Contracts**: `core/app_model.py` is the single source of truth for all
  dataclasses shared by API ⇄ clients.
- **Recommendations** (`core/recommendation/`): health-scenario rules +
  community/feedback blend; `ml_ranker.py` activates at ≥20 sessions.
- **Clients**: `frontend/` React+Vite (works against the real API) — bilingual
  he/en with RTL (`src/i18n.tsx`, no deps), light "clinical" design system
  (tokens in `index.css`: paper bg, black primary buttons, pine-green accent),
  top-nav tabs: Home / Workout (placeholder) / History.
  `mobile/` Expo RN scaffold (fake data, frozen — web won over RN).

## Commands

```bash
python -m pytest tests/ -q          # full suite (~277 tests, mongomock)
python -m core.exercise.seed        # seed exercises into Mongo (idempotent)
python -m core.ml.trainer           # retrain phase models + write exercise_angles
python -m core.ml.eval "data/labeled_vidz/squat/<file>.json"   # held-out eval
```

Setup order on a fresh DB: seed → trainer → register a user → pipeline.
Demo user in the local dev DB: `demo@poseiq.dev` / `demo1234`.

## Critical implementation notes

- **Two angle spaces.** `core/ml/angles.py` computes 12 angles (knee/hip/elbow/
  shoulder/ankle) in *normalized 2D* — the space models are trained in. The
  legacy `core/detection/angle_calculator.py` computes 8 angles (incl. spine,
  legs_spread) in *3D pixels*. The live pipeline merges both dicts with **ML
  angles winning on shared names** (`angles.frame_from_live` adapter). Never
  feed legacy 3D knee/elbow values to the models or Mongo ranges.
- **Feature vector v2** (`trainer.FEATURE_VERSION`): 12 angles + 12 per-joint
  rolling velocities in **degrees/second** (wall-clock normalized). Bundles
  carry `feature_version`; `classify_trained` handles v1 (13-feat) bundles too.
  Live velocity comes from `classifier.DeltaTracker` (pass `now=time.time()`).
- **Temporal smoothing**: `core/ml/smoother.PhaseSmoother` — majority vote +
  legal phase-cycle constraint + resync. Used in eval and the live overlay.
- **The seed JSON has no angle ranges.** `ExerciseModel()` from JSON yields
  phases with empty `angles` — the state machine can't run on it. The pipeline
  loads `ExerciseModel.from_mongo()` and errors clearly if trainer hasn't run.
- **Non-ASCII project path** ("פרויקט גמר") breaks MediaPipe's C++ layer and
  cp1252 consoles. Workarounds already in place: models cached in `~/.pose-iq`,
  chdir around landmarker creation (extractor + pose_detector), and
  `sys.stdout.reconfigure(errors='replace')` in ML CLIs. Keep these patterns.
- **Auth**: scrypt password hashing (stdlib) with transparent upgrade of
  legacy sha256 on login; `require_auth` binds the token to the route's
  user_id (403 otherwise). Tests cover both — don't regress.
- Training videos (~190MB) are **not in git**; label JSONs reference
  `data/videos/{ex}/good/*.mp4`.

## ML state (2026-07-14)

Held-out accuracy after velocity features + smoothing (commit 49efb8c):
frame-exact raw ~53% avg (squat 63, lunge 61, biceps 56, shoulder_press 32);
**±0.3s boundary-tolerant: squat 81 / lunge 87 / biceps 89 / shoulder_press 53**.
Conclusion: **data-limited** — only 2-3 training videos per exercise,
different sources. Next lever: user films videos per
`docs/filming-protocol-he.md` (people/angles/labeling rules + bad-form clip
list). User targets 98-99% — realistic on rep-level metrics once data grows.
shoulder_press is inherently hard (slow eccentric ≈ static start position).

**Rep-level metric** (`core/ml/reps.py`, wired into eval.py, `--rep-tol`):
a rep anchors at the transition into the cycle's final phase (turnaround);
truth↔pred anchors matched greedily within ±0.5s; reports matched/missed/
extra + recall/precision/F1 for raw and smoothed. Baseline on all 13 labeled
videos (train+test mixed → optimistic): smoothed rep recall lunge 91 /
squat 83 / shoulder_press 70 / biceps 50 — biceps has only 2 complete labeled
reps total; several label files are partial (eccentric-only, setup footage).
This is the headline metric for the report's accuracy-vs-data curve.

**Weight tracking + progressive overload** (2026-07-14): sessions carry
optional `weight_kg` (pipeline 3rd CLI arg; `PUT /api/user/<id>/sessions/
<sid>/weight`; editable in frontend session cards). Double-progression rules
in `core/recommendation/overload.py` gated on form score (≥85 clean ×2-3
sessions → +increment; <70 → back off), surfaced via `GET .../weights` and
embedded in the dashboard (`weight_recommendations`).

## Roadmap (agreed with user)

1. ~~Fix broken pipeline, auth holes, retrain~~ (done, 2c81b92)
2. ~~Close the loop: pipeline → Mongo → dashboard~~ (done, 6b5bc94 + tests)
3. ~~Velocity features + smoothing + honest eval~~ (done, 49efb8c)
4. **More labeled videos** → accuracy curve for the report. Rep-level metric
   ~~built~~ (core/ml/reps.py); filming protocol written
   (docs/filming-protocol-he.md) — waiting on user to film
5. Web client with in-browser pose (MediaPipe Tasks JS) — decided web over RN
6. ~~Weight tracking + progressive-overload recommendations~~ (done,
   overload.py + API + frontend, 2026-07-14)
7. Cloud: Mongo Atlas + API on Render/Railway + frontend on Vercel + CI
8. Bad-form clips → binary quality classifier (planned; filming protocol
   already covers bad-form clips with per-mistake tags)

`docs/pose-iq-status-he.md` is the Hebrew status doc for the advisor — it
predates the Mongo migration and ML rewrite; update it before any submission.
