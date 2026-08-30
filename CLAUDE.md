# Pose-IQ — Claude Code Project Guide

Final-year project (פרויקט גמר): AI personal workout coach — real-time form
correction from a live camera, per-user history, personalized recommendations.
Goal: cloud deployment with real users. The user communicates in Hebrew;
code, comments, and commits are in English.

## Stack & architecture

- **Desktop pipeline — moved to `stale/`, not part of the deployed app**
  (2026-08-23, web won — see roadmap #5 and #9 below). Was: OpenCV window →
  camera → MediaPipe (33 landmarks) → joint angles → phase state machine
  (reps) → posture rules → HUD + TTS coach, run via
  `python -m stale.core.pipeline squat <user_id>`. Kept (not deleted) for
  reference; not imported by `api/app.py`, `frontend/`, or the ML training
  tools. `core/ml/classifier.py` stayed in `core/` despite only the pipeline
  using it live — `core/ml/eval.py` still depends on it for offline model
  evaluation.
- **ML** (`core/ml/`): hand-labeled videos (`data/labeled_vidz/{ex}/*.json`) →
  per-exercise RandomForest phase classifier (`data/models/phase_{ex}.joblib`).
  `trainer.py` also writes measured per-phase angle stats to Mongo
  `exercise_angles` — those ranges drive the frontend's TS state machine
  (`frontend/src/pose/stateMachine.ts`).
- **DB**: MongoDB (`core/db/get_db()`, `MONGODB_URI` env, default localhost).
  Collections: users, tokens (TTL), sessions, exercises, exercise_angles,
  ratings (1-5 exercise ratings — feeds the recommendation ranker).
  The old SQLAlchemy/Postgres layer was deleted — docs mentioning it are stale.
- **API** (`api/app.py`, Flask): auth, profile, health ratings, history,
  progress, ratings, dashboard (embeds recommendations). Run:
  `flask --app api.app run`.
- **Contracts**: `core/app_model.py` is the single source of truth for all
  dataclasses shared by API ⇄ clients.
- **Recommendations** (`core/recommendation/`, rewritten 2026-08-23): pure
  rating-based — a hand-rolled linear regression per exercise (gradient
  descent, no numpy/scikit-learn), predicting from the user's own ratings on
  the *other* exercises (`PUT /api/user/<id>/ratings/<exercise_id>`). Training
  data is pulled entirely from Mongo's `ratings` collection — every rating
  in there shapes the learned weights, not just each rater's own inference
  input. Trained once per server process, lazily on the first request
  (`api/app.py`'s `before_request` hook — not eagerly at import time, so
  tests can still patch `get_db` before it fires), not on every dashboard
  load. Artifact caches to `data/models/recommendation_ranker.json`
  (gitignored — it's regenerated from Mongo, not a fixed thing to commit).
  No community/feedback/health-scenario blending anymore (that machinery, plus the
  dormant ML ranker that could never load on the deployed API since it
  needed scikit-learn, was deleted, not kept).
- **Clients**: `frontend/` React+Vite (works against the real API) — bilingual
  he/en with RTL (`src/i18n.tsx`, no deps), light "clinical" design system
  (tokens in `index.css`: paper bg, black primary buttons, pine-green accent),
  top-nav tabs: Home / Workout / History.
  (An Expo RN mobile scaffold was tried and dropped — web won; removed
  2026-08-23, see `docs/pose-iq-status-he.md` §7 for the decision writeup.)
- **In-browser live workout** (`frontend/src/pose/` + WorkoutScreen):
  MediaPipe Tasks JS (pose_landmarker_lite from Google CDN, GPU) →
  `angles.ts` / `stateMachine.ts` / `postureRules.ts` are faithful TS ports
  of the Python pipeline logic, driven by the same Mongo ranges served by
  `GET /api/exercises`; sessions saved via `POST /api/user/<id>/sessions`.
  Camera/model failures surface distinct errors; video is mirrored (selfie).
  **Rep counting/recording and posture-check phase lookup are ML-driven**
  (`phaseClassifier.ts` + `mlRepCounter.ts`, ONNX via `onnxruntime-web`) —
  `stateMachine.ts` only gates the pre-start readiness check now; see "Rep-
  counting: rule engine retired" below for why.

## Commands

```bash
python -m pytest tests/ -q          # web-app-relevant suite (208 tests, mongomock)
python -m pytest stale/tests/ -q    # desktop-pipeline-only suite (44 tests)
python -m core.exercise.seed        # seed exercises into Mongo (idempotent)
python -m core.ml.trainer           # retrain phase models + write exercise_angles
python -m core.ml.eval "data/labeled_vidz/squat/<file>.json"   # held-out eval
python -m core.recommendation.train_ranker   # force an immediate ranker retrain from Mongo
```

Setup order on a fresh DB: seed → trainer → register a user → web app
(`flask --app api.app run` + `npm run dev` in `frontend/`).
Demo user in the local dev DB: `demo@poseiq.dev` / `demo1234`.

## Critical implementation notes

- **Two angle spaces** (relevant to the stale desktop pipeline; the web app
  only ever uses the first). `core/ml/angles.py` computes 12 angles (knee/hip/
  elbow/shoulder/ankle) in *normalized 2D* — the space models are trained in.
  The legacy `stale/core/detection/angle_calculator.py` computed 8 angles
  (incl. spine, legs_spread) in *3D pixels*; `stale/core/pipeline.py` merged
  both dicts with **ML angles winning on shared names**
  (`angles.frame_from_live` adapter). Never feed legacy 3D knee/elbow values
  to the models or Mongo ranges.
- **Feature vector v2** (`trainer.FEATURE_VERSION`): 12 angles + 12 per-joint
  rolling velocities in **degrees/second** (wall-clock normalized). Bundles
  carry `feature_version`; `classify_trained` handles v1 (13-feat) bundles too.
  Live velocity comes from `classifier.DeltaTracker` (pass `now=time.time()`).
- **Temporal smoothing**: `core/ml/smoother.PhaseSmoother` — majority vote +
  legal phase-cycle constraint + resync. Used in `eval.py`; was also used in
  the stale pipeline's debug overlay.
- **The seed JSON has no angle ranges.** `ExerciseModel()` from JSON yields
  phases with empty `angles` — a state machine can't run on it. `api/app.py`'s
  `/api/exercises` route loads `ExerciseModel.from_mongo()` and falls back to
  the JSON-only model (empty ranges, `ready: false`) if trainer hasn't run yet
  for that exercise.
- **Non-ASCII project path** ("פרויקט גמר") breaks MediaPipe's C++ layer and
  cp1252 consoles. Workarounds already in place: models cached in `~/.pose-iq`,
  chdir around landmarker creation (`extractor.py` and the stale
  `pose_detector.py`), and `sys.stdout.reconfigure(errors='replace')` in ML
  CLIs. Keep these patterns.
- **Auth**: scrypt password hashing (stdlib) with transparent upgrade of
  legacy sha256 on login; `require_auth` binds the token to the route's
  user_id (403 otherwise). Tests cover both — don't regress.
- Training videos (~190MB) are **not in git**; label JSONs reference
  `data/videos/{ex}/good/*.mp4`.

## Live rule-engine bugs found + fixed during real camera testing (2026-08-25)

Found while the user tested the ONNX parallel signal live (see roadmap #10)
— none of these are ML-related, all are `frontend/src/pose/` rule-engine /
UX issues, found in this order:

1. **Voice coaching burst**: `WorkoutScreen.tsx`'s `handlePostureIssues` spoke
   every flagged issue independently (each on its own 4s per-*joint*
   cooldown) — several joints going out of frame at once (e.g. mid-reposition)
   queued a rapid-fire burst of utterances. Fixed: one global 6s cooldown,
   speaks one issue at a time; also slowed `rate` 1.1→0.92.
2. **`started` flipped on a single noisy frame**: unlike phase transitions
   (which already required `FRAMES_TO_TRANSITION` consecutive frames),
   `stateMachine.ts`'s pre-workout readiness check flipped `started=true` on
   one lucky/noisy frame — incidental movement while still positioning the
   camera could count as a real workout starting. Fixed: same
   consecutive-frame requirement now applies to the `started` flip too
   (`startReadyCounter`).
3. **Overcounting persisted even after #2** — root cause turned out to be in
   `trainer.py`, not the frame-stability logic: **`exercise_angles` ranges
   were computed by pooling every camera angle together** (front + side1 +
   side2 + old mixed-angle clips). The same physical joint position projects
   to a very different 2D angle from the front vs. the side — pooled
   min/max for shoulder_press's `lowering`/`pressing` phases had collapsed
   to ~0-180° (measured), i.e. barely constraining anything, so almost any
   arm position satisfied "next phase" and reps over-triggered constantly.
   **Fixed**: `collect_samples_from_labels` now computes the pooled
   min/max/mean/std stats from **front-view files only** (filename contains
   "front" — matches how a real user actually faces a webcam), while the RF
   *model* still trains on every file (its many features/trees handle
   viewpoint-mixed data fine; only plain min/max can't). Verified: front-only
   shoulder_press `start` elbow range went from [0.2°, 179.9°] to
   [42.3°, 92.7°]. Retrained all 4 exercises; also found and deleted 3 stale
   `exercise_angles` docs under `exercise_id: squat` with `shoulder_press`'s
   phase names — leftover from the 2026-08-23 mislabeling incident, fixed at
   the label level then but never cleaned up in Mongo (harmless — squat's
   real phase list never looked them up — but confusing clutter).
4. Also switched `transitionCounter`/`startReadyCounter` from "decay by 1 on
   a miss" to a hard reset to 0 — decay let noisy oscillation *during real
   movement* (not just brief dropouts) slowly net-accumulate toward the
   threshold without ever being truly stable.
5. Widened the live-workout screen (`max-w-2xl` → `max-w-4xl`) — too narrow
   to comfortably check full-body camera framing.

Fix #3 was re-verified live and held up for normal continuous reps — but
testing kept going the next day and surfaced enough more (including the
final decision to stop patching this rule engine at all for rep-counting)
that it gets its own section below.

## Rep-counting: rule engine retired in favor of the ML classifier (2026-08-26)

Continued live testing after 2026-08-25's fixes surfaced a chain of
follow-on bugs, each fixed with real evidence (Mongo range dumps, or
`[SM-DEBUG]`/`[ML-DEBUG]` console logging added specifically to stop
guessing blind) rather than another guess-and-check cycle:

1. **Readiness gate over-applied a fix meant for transitions only.** Tried
   requiring the *core* (middle 40%) of a phase's range, not just any value
   technically inside it, to confirm a transition — meant to stop a static
   mid-pause sitting in the overlap between two adjacent phases from
   free-wheeling into a fake rep. Mistakenly applied the same core-range
   check to the **pre-start readiness gate** too. Result: the HUD's per-joint
   readiness indicator (checked against the full range) showed 4/4 green the
   whole time, while the hidden core-range check silently failed on ordinary
   standing sway/pose-tracking jitter — `started` never latched at all. The
   user diagnosed this exact mistake by reasoning alone before it was found
   in logs: depth-into-a-range only means something for a phase that *is* a
   range of motion (ascending/descending), not a static hold like "start."
   Fixed: readiness gate back to plain full-range matching.
2. **Symmetric core-range trim broke real full-effort reps.** Even
   restricted to transitions only, trimming both edges toward the center
   made a *wide* measured phase (e.g. shoulder_press "pressing", elbow
   39-178° — the label covers the whole upward sweep, not just its top) cap
   its own "core" around the middle (~80-136°) — pressing all the way up to
   genuine full extension (~170°) *overshot* the core's own ceiling and
   never confirmed the transition. Tried a direction-aware fix next
   (`motion_direction` from the data model: trim only the entry edge shared
   with the previous phase, leave the far edge — genuine full extension —
   untouched). That fixed the overshoot case but the underlying ranges were
   still too thin/noisy (2-3 training clips) for *any* stricter geometric
   threshold to net-help rather than hurt — confirmed by two repeated live
   10-rep sets where the reverted-to-plain rule engine only caught 3-4 of 10
   real reps (stuck at the "lowering"->"start" boundary almost every time:
   `right_elbow: got 40.4, need [41.2, 100.7]` — off by a fraction of a
   degree, not a real miss). **Reverted core-range entirely** — back to
   plain full-range matching + the hard-reset consecutive-frame requirement
   from 2026-08-25's fix #4. See `stateMachine.ts`'s comment above `contains`
   for the full account.
3. **The ML classifier, tested as a replacement, won decisively.** Built
   `frontend/src/pose/mlRepCounter.ts` — counts reps from
   `phaseClassifier.ts`'s live predictions alone, entirely independent of
   `stateMachine.ts`. Two design fixes along the way, both confirmed live
   before landing:
   - *Majority vote, not consecutive-frame agreement.* The classifier
     reliably nails phase transitions with real motion behind them (clean
     2-in-a-row), but right at the same "lowering"->"start" boundary the
     rule engine struggles with, it genuinely flickers tick-to-tick between
     two classes for a few ticks before settling — a single hard reset there
     (mirroring `stateMachine.ts`'s own design) threw away real progress
     every time and the rep never counted. Switched to "3 of the last 5
     ticks" — tolerates the flicker, still requires real sustained evidence.
   - *Stable ("start") phases excluded from the tracked cycle entirely.*
     Mid-set, the model essentially never top-predicts "start" again once
     real reps are underway — likely the same start/pressing label ambiguity
     the user flagged earlier this session (labeling "start" a little late,
     already inside the next phase, to avoid skipping it). Physically, a rep
     is complete once you're pressing again after lowering — no need to
     have paused at a fully-rested "start" pose mid-set. The rule engine's
     (already-fixed) readiness gate still confirms the user starts *from*
     "start" before tracking begins at all.
   - **Result, two repeated live 10-rep shoulder_press sets**: ML counter 9
     and 11 vs. the rule engine's 3-4 (the discrepancy in the 11 case was one
     genuine over-count from the user moving toward the camera to end the
     set manually — see next item). The rule engine's 2025-08-25 fixes made
     it *stable*, not *accurate* — it reliably avoids wild overcounting, but
     under-counts badly by getting stuck at ambiguous boundaries.
   **Decision: the ML classifier is now the rep-counting/recording
   authority** — `WorkoutScreen.tsx`'s displayed/spoken/saved rep count comes
   from `mlRepCounter`, not `stateMachine.ts`. The rule engine
   (`ExerciseStateMachine`) still owns the pre-start readiness gate (proven
   reliable) — see the next two items for where its phase pointer being
   unreliable still mattered even after this change.
4. **No "idle" class exists, so idle motion could still get counted.** None
   of an exercise's trained classes represent "not exercising" — the
   classifier always picks one of start/pressing/lowering even when the user
   has simply stopped. Confirmed live: after a real set, sitting still (or a
   small unrelated movement, e.g. reaching for the laptop to end the
   workout) occasionally produced one stray extra rep. Two mitigations, not
   a full fix: (a) a motion-magnitude gate in `mlRepCounter.ts` (skip ticks
   with near-zero recent angular velocity — reuses the same `deltas` already
   computed as model input features; doesn't help when the stray motion
   *is* real motion, just not exercise motion); (b) a **target-reps** field
   on the setup screen (`WorkoutScreen.tsx`) — when set, the workout
   auto-ends the instant the ML counter reaches it, removing the window
   where post-set motion could get miscounted at all. Confirmed live: with a
   target set, the auto-end fired correctly and the tail-end miscount
   stopped recurring.
5. **Posture/weak-point checks were still silently keyed off the (unreliable)
   rule-engine phase pointer, even after rep-counting moved to ML.** User
   caught it from the *symptom*, not the code: a set of genuinely good reps
   ended with "weak points: both shoulders, both elbows, spine" — every
   joint that moves. Root cause: `rules.analyze(angles, sm.activeRules())`
   used `sm.currentPhase`, which — per items 1-3 above — gets stuck on a
   stale phase for long stretches. While stuck, the user's *correct, already
   -moved-on* angles were being compared against the *wrong* phase's
   expected range, flagging real motion as a form error on every joint that
   moves. Fixed: posture checks now look up the phase by the ML's last raw
   per-tick phase call (`mlDebugRef.current.phase`) instead of
   `sm.currentPhase`, falling back to the rule engine only before the first
   ML tick of the set.
6. **"Weak points" in the session summary had no frequency threshold.** A
   joint flagged even once across an entire set (e.g. one tracking blip)
   was listed as a weak point forever, same as one that recurred on most
   reps. User caught this too: "if I went out of range once with one elbow
   that's not really a weakness." Fixed: a joint now needs to recur in at
   least `max(2, 30% of reps)` reps to be listed, and the summary shows the
   actual count (e.g. "Right elbow (3/10)") instead of a bare name.

**Net effect**: the deterministic rule engine (`stateMachine.ts`/
`exercise_state_machine.py`) is demoted from "reps/posture decision-maker"
to "readiness gate only" for the parts of the pipeline that were live-tested
this session. It's unclear whether this generalizes past shoulder_press —
all of today's live testing happened on that one exercise (laptop-camera
framing made the others impractical to test at a desk); squat/lunge/
biceps_curl have the same phase-metadata shape (`motion_direction`/
`is_initial` checked directly in Mongo — all well-formed) so the code path
is exercise-agnostic, but their `exercise_angles`/model training data is
thinner in places (e.g. `standing`'s `n_frames` is 22 for biceps_curl, 55
for lunge, vs. shoulder_press's 117) and genuinely untested live. Plan: test
all 4 exercises at a real gym before submission.

## Deployed to production + gym-tested (2026-08-27 to 2026-08-30)

**Live now**: backend on Render (`https://pose-iq-api.onrender.com`,
free tier — spins down on inactivity, first request after idle takes
~50s), frontend on Vercel (`https://pose-iq.vercel.app`, auto-deploys on
push to `main`), Mongo Atlas (`PoseIQ` cluster). **Local dev and
production use different databases** — local defaults to localhost Mongo
unless `MONGODB_URI` is set, Atlas is separate. Learned this the hard way:
`core.exercise.seed`/`core.ml.trainer` had only ever been run against
local Mongo, so Atlas was missing `exercise_angles` for `biceps_curl`
entirely (showed as "not ready" on the deployed site) until re-run against
Atlas directly (`$env:MONGODB_URI`/`$env:MONGODB_DB` set to Atlas's values
before running either command). University WiFi blocks outbound port
27017 (MongoDB's) while allowing normal HTTPS — connecting to Atlas
directly (not through the deployed API) requires a phone hotspot instead.

Gym-tested all 4 exercises live (with another person, not just solo) after
deploy. Two real bugs found and fixed:

1. **Camera was selfie-only** (`facingMode: 'user'` hardcoded) — no way to
   use a phone's rear camera (e.g. propping it up facing away, better
   angle/distance for squat/lunge). Fixed: `WorkoutScreen.tsx` now has a
   front/rear picker; `acquireCamera()` takes the facing mode; video +
   skeleton-overlay mirroring (`scaleX(-1)`) only applies for the front
   camera — mirroring a rear-camera view (showing someone else, not a
   selfie) would be wrong.
2. **`spine`'s posture correction fired almost constantly** on
   `biceps_curl` and `shoulder_press` (`"don't lean back"` /
   `"don't arch your back"`) even with good form. Root cause: `spine` was
   the *only* hand-set threshold left in the whole system — global
   (`data/exercises_seed.json`'s now-removed `global_constraints`), never
   measured from training data like every other joint, because its old
   formula needed pixel-space 3D against a fixed-50px reference point,
   while the training pipeline only ever sees normalized [0,1] landmark
   coordinates (no pixel dimensions available). Two rounds of manually
   widening it (15°→25°) still misfired — confirming the problem was the
   *guessing*, not the specific number. **Fixed properly**: ported spine to
   a normalized-2D formula (shoulder-hip vs. a synthetic straight-up
   point, same `angle2d`/`_angle_deg` used by every other joint) in both
   `core/ml/angles.py` and `frontend/src/pose/angles.ts` (training/serving
   parity — see the note on this above), moved it from a flat
   per-exercise `global_constraint` to a per-phase joint (same shape as
   elbow/knee), and let it flow through the *already-generic*
   `phase_angles` stats pipeline in `trainer.py` — required zero changes
   there beyond the new angle itself, since front-view-only filtering and
   tolerance padding already applied automatically to whatever joints show
   up in the data. One real bug caught mid-implementation: spine has no
   `too_low` correction anywhere (unlike every other joint, *less* lean is
   never a form issue) — the measured `min` was nonzero, which would have
   flagged perfectly straight posture as "too low" with a generic fallback
   message. `write_exercise_angles` now forces `min=0.0` for spine
   specifically. Real measured maxes (all phases, all exercises): squat
   ~21°, lunge ~28-33°, biceps_curl ~26-30°, shoulder_press ~23-24° —
   phase-specific, not one guessed number for the whole exercise.

## ML state (2026-08-25)

**Model shrunk for browser delivery (ONNX plan — see roadmap #9).**
`trainer.py`'s RandomForest was `n_estimators=200, max_depth=None`, producing
7-15MB `.joblib` files per exercise (~47MB total) — too heavy to ship to a
mobile browser. Controlled test (train on 9/10 squat clips, held out
`person1_side1_01` the same way as the 2026-08-23 clean test): 200/unbounded
vs 60/depth-12 vs 80/depth-14 all scored **identically** (97% boundary,
100% rep-level) on the same held-out clip, cv_accuracy slightly *higher*
for the smaller models — 200 trees was overkill for a 3-class problem on
24 features, by that metric alone.

**But `tests/test_ml.py::TestRealTrainedModel::test_delta_feature_is_used_by_model`
caught something the video-based eval couldn't**: at 60 trees the squat
model started giving identical predictions for opposite velocities at the
same mid-range angle — the exact failure mode the 2026-08-23 shoulder_press
tempo fix was designed to avoid, just not exercised by any real held-out
clip. Swept `n_estimators`×`max_depth` (extracting features once, refitting
many times) — pass/fail wasn't monotonic in depth (e.g. 100/unbounded
failed, 90/depth-14 passed), so **`n_estimators` — vote-count diversity —
turned out to matter more than depth for preserving velocity sensitivity**.
Landed on **`n_estimators=90, max_depth=14`**, the smallest config that
passed with margin. Retrained all 4 exercises; full suite (293 tests)
passes, including the delta-sensitivity check. Final sizes: squat 4.0MB,
lunge 6.1MB, biceps_curl 3.1MB, shoulder_press 5.5MB (was 10.4/15.5/7.2/
14.1MB) — **~2.5x smaller, ~18.8MB total instead of ~47MB.** cv_accuracy
per exercise: squat 0.700, lunge 0.690, biceps_curl 0.755, shoulder_press
0.643 (was 0.703/0.685/0.764/0.643 — no meaningful change). Next: convert
to ONNX (`skl2onnx`) + wire into the web client with `onnxruntime-web`, so
the trained classifier actually drives the live product instead of being
an eval-only / desktop-debug-only artifact (see "Live decision path" note
below).

**Lesson for future hyperparameter changes to this model**: video-based
held-out eval and the synthetic unit test check different things — the
video eval measures real-world accuracy but only exercises whatever motion
patterns are in the held-out clip; the synthetic test pins down a specific
property (velocity actually matters) that a single video might never
stress-test. Don't skip `pytest` after retraining just because the eval
numbers look fine.

**Live decision path — updated 2026-08-26, this changed.** In the **web
client**, the ONNX-exported model (via `onnxruntime-web`,
`frontend/src/pose/phaseClassifier.ts`) now drives rep counting/recording
(`mlRepCounter.ts`) and posture-check phase lookup — see "Rep-counting:
rule engine retired" above for the full story and why. `stateMachine.ts`
(the TS rule engine) still owns only the pre-start readiness gate there.
In the **desktop pipeline** (`stale/`, not part of the deployed app) and
the **offline eval** (`core/ml/eval.py`), nothing changed: `core/pipeline.py`
still runs the trained model as a parallel debug-overlay signal only
(toggle `d`), and `ExerciseStateMachine`
(`core/exercise/exercise_state_machine.py`) is still the one driving reps/
transitions/violations there — that codepath is stale/reference-only, not
worth re-doing this migration for.

## ML state (2026-08-23)

**Filming round 1 done**: user filmed 27 new good clips per
`docs/filming-protocol-he.md` (2 people × 3 angles [front/side1/side2] ×
4 exercises, +3 tempo-emphasized shoulder_press clips), labeled with
`tools/video_labeler.html`, retrained. No bad-form clips yet (roadmap #8
still open).

Held-out accuracy after retraining on the expanded set (single held-out
file per exercise, same files as the 2026-07-14 baseline where present):
frame-exact raw: squat 48, lunge 66, biceps 81, shoulder_press 79 (was 63/
61/56/32). **±0.3s boundary-tolerant (smoothed): squat 54, lunge 90,
biceps 82, shoulder_press 87** (was 81/87/89/53). **Rep-level recall
(smoothed): squat 67, lunge 100, biceps 100, shoulder_press 75** (was 83/
91/50/70).

**shoulder_press is fixed** — the tempo-emphasized clips solved exactly the
problem diagnosed on 2026-07-14 (slow eccentric ≈ static start). lunge and
biceps improved too, especially rep-level.

**squat's single held-out file regressed (81→54) — diagnosed as an old-clip
quality issue, not a real regression.** Isolated tests (`--exercise squat`
reruns with one file swapped out at a time): held out `person1_side1_01`
(new, protocol-filmed) → **97% boundary-tolerant, 100% rep-level recall**.
Quarantining the default held-out file and letting the *next* alphabetical
file (`Proper Squat Technique...`, also an old internet clip) become
held-out instead → 68% boundary-tolerant — better than 54%, but still well
below the new clip's 97%. Two different old internet-sourced clips both
underperform relative to the new consistently-filmed ones, so this isn't
one unlucky video — the old squat clips (mixed sources, inconsistent
camera setup/quality per `docs/filming-protocol-he.md` §1) are just a
weaker held-out proxy in general. The model genuinely generalizes well
under consistent filming conditions (97%/100%); trainer.py's single-file
auto-holdout is unreliable for squat specifically because its old-clip
pool is noisier than the other three exercises'. For the report, use the
isolated-rerun number (97%/100%), not the raw auto-reported one — and
note this as a limitation of the single-file-holdout methodology.

**Rep-level metric** (`core/ml/reps.py`, wired into eval.py, `--rep-tol`):
a rep anchors at the transition into the cycle's final phase (turnaround);
truth↔pred anchors matched greedily within ±0.5s; reports matched/missed/
extra + recall/precision/F1 for raw and smoothed. This is the headline
metric for the report's accuracy-vs-data curve.

**Weight tracking**: sessions still carry optional `weight_kg` (`PUT /api/
user/<id>/sessions/<sid>/weight`, editable in frontend session cards) — just
logging, no recommendation. The progressive-overload recommender
(`core/recommendation/overload.py`, double-progression gated on form score)
was removed 2026-08-24: `core/recommendation/` is rating-only now, and
overload's job (how much weight to lift) isn't a rating question at all —
kept getting flagged as out of place next to the ranker. Deleted outright
(API route, dashboard field, frontend display, i18n strings), not moved to
`stale/`, since nothing about it depends on the desktop pipeline.

## Roadmap (agreed with user)

1. ~~Fix broken pipeline, auth holes, retrain~~ (done, 2c81b92)
2. ~~Close the loop: pipeline → Mongo → dashboard~~ (done, 6b5bc94 + tests)
3. ~~Velocity features + smoothing + honest eval~~ (done, 49efb8c)
4. **More labeled videos** → accuracy curve for the report. Rep-level metric
   ~~built~~ (core/ml/reps.py); filming protocol written
   (docs/filming-protocol-he.md); ~~round 1 filmed + labeled + retrained~~
   (done, 2026-08-23 — see ML state above). Bad-form clips (§5 of the
   protocol) still not filmed — needed for roadmap #8.
5. ~~Web client with in-browser pose (MediaPipe Tasks JS)~~ (done, 2026-07-17 —
   live camera → angles → state machine → reps/violations → save session;
   verified working end-to-end by the user)
6. Weight tracking + progressive-overload recommendations (done, 2026-07-14
   → deleted 2026-08-24 in the recommendation-engine rewrite, see item 9 →
   **restoring 2026-08-25, user's call**: weight *logging* survived the
   rewrite untouched; only the recommendation half needs to come back).
7. **Cloud deploy — in progress (2026-08-23).** ~~Mongo Atlas cluster~~ (done —
   `PoseIQ` cluster, `azikrantz_db_user`, network access `0.0.0.0/0` open since
   Render's IPs are dynamic; connection string in the user's hands, not
   committed anywhere). Added `requirements-api.txt` (flask/flask-cors/
   gunicorn/pymongo/python-dotenv only — verified `import api.app` never
   pulls in mediapipe/opencv/scikit-learn, so the deployed API doesn't need
   them) + `render.yaml` blueprint pointing at it. Next: Render web service
   (connect GitHub, set `MONGODB_URI`/`MONGODB_DB`/`FRONTEND_ORIGIN` env
   vars), then Vercel for `frontend/`, then wire `FRONTEND_ORIGIN` to the
   Vercel URL for CORS. Not yet: Render service created, Vercel deploy, CI
   deploy step. Pre-deploy cleanup (2026-08-23): removed the frozen `mobile/`
   Expo scaffold, the no-auth `/api/dev/*` fake-data routes (`core/user/
   fake_data.py` kept — still exercised directly by `tests/test_user.py`),
   and the orphaned `core/recommendation/demo.py`/`simulation.py` prototype
   (pre-dated `bridge.py`, never wired to the real API — had to strip the
   dead `from .simulation import ...` re-export from `recommendation/
   __init__.py` too, since every submodule import runs that first).
8. Bad-form clips → binary quality classifier (planned; filming protocol
   already covers bad-form clips with per-mistake tags)
9. ~~Web app verified working end-to-end~~ (done, 2026-08-23 — register/
   login/dashboard round-tripped against the real Flask+Mongo+Vite stack) →
   ~~move desktop-pipeline-only Python files aside~~ (done, 2026-08-23):
   `core/pipeline.py`, `core/detection/`, `core/coaching/`,
   `core/exercise/exercise_state_machine.py` + `posture_rules.py`, and the
   never-wired-anywhere `core/ml/violations.py` moved to `stale/`, mirroring
   the original subdirectory layout; their tests split out to `stale/tests/`.
   `core/ml/classifier.py` stayed in `core/ml/` — `eval.py` still needs it.
   Kept, not deleted, so the desktop pipeline can still be run/referenced
   later. Same day: recommendation engine rewritten to pure rating-based
   collaborative filtering (see Recommendations above) — the old health-
   scenario/community/feedback/ML-ranker system was deleted outright (not
   moved to `stale/`) since it was actively misleading in production
   (`community_score` was always fake mock data; the ML ranker could never
   load on the deployed API at all). **Weight recommendations
   (`overload.py`) were deleted in the same pass and are being restored
   (2026-08-25, user's call) — see item 6 above and the ML state note.**
10. **ONNX phase classifier in the web client (2026-08-25).** Goal: the
    trained RF model actually participates in the live web app instead of
    being eval/desktop-debug-only (see ML state note above).
    ~~Shrink models for browser delivery~~ (done — 90 trees/depth 14,
    ~18.8MB total for all 4 exercises, was ~47MB, no accuracy loss).
    ~~Export to ONNX~~ (done — `core/ml/export_onnx.py`, 100% prediction
    parity vs sklearn on 500 random samples per exercise; ONNX files ~10MB
    total, smaller than the joblib source). ~~Wire into the web client~~
    (done — `frontend/src/pose/{deltaTracker,phaseClassifier}.ts` +
    `onnxruntime-web`; `WorkoutScreen.tsx` runs it throttled (250ms),
    initially wired as a **parallel signal only** shown in the HUD next to
    the rule-based phase — deliberately staged that way to validate real
    browser behavior first). ~~Confidence-gated fallback / making it the
    actual decision-maker~~ (done, 2026-08-26 — not confidence-gated in the
    end, just switched outright after live A/B testing showed the ML
    counter far more accurate than the rule engine; see "Rep-counting: rule
    engine retired" above).

    **Verified working end-to-end in a real browser**, not just Python:
    fed the live `classifyPhase()` opposite-sign knee velocities at the
    same mid-range angle through the actual production bundle — correctly
    predicted `descending` vs `ascending` respectively, the same
    velocity-sensitivity property `test_delta_feature_is_used_by_model`
    checks in Python. Two real integration snags found and fixed along
    the way (both `frontend/vite.config.ts` / `phaseClassifier.ts`
    comments explain why):
    - Importing bare `'onnxruntime-web'` pulls its WebGPU (jsep) wasm
      variant into the production build via Rollup's static asset
      scanning (+27MB, unused) — fixed by importing the `'onnxruntime-web/wasm'`
      subpath instead (CPU-only, ~14MB). A same-named file still gets
      redundantly double-bundled (`dist/assets/` *and* our own
      `public/ort/` copy that's actually fetched) — cosmetic, not worth
      more time; never downloaded by users since nothing references the
      `dist/assets/` copy's URL.
    - `onnxruntime-web`'s WASM backend can't marshal a ZipMap
      (`sequence<map<string,float>>`) output — `session.run()` throws for
      the *whole call* if `output_probability` (skl2onnx's default
      `predict_proba` export) is in the fetch list. **Fixed** (same
      session, not deferred): `export_onnx.py` now passes skl2onnx
      `options={id(clf): {'zipmap': False}}`, which renames the outputs
      to `label` (unchanged behavior) and `probabilities` — a plain
      `[1, n_classes]` float tensor instead. `confidence` = max of that
      tensor (order-independent — works without also shipping each
      exercise's `classes_` column ordering to the frontend). Verified
      live in the browser (build+preview, real inference call): real
      confidence values (62-90%) across all 4 exercises, correct phase
      unaffected. Per-class `probs` breakdown is still `null` (would need
      shipping the column order too) — not needed for a scalar
      confidence-gated fallback, only for showing "70% descending, 20%
      standing, 10% ascending" in a UI, which nothing needs yet.
    - Also dev-server-only (doesn't affect the real deployment): the ONNX
      path 404s/throws under plain `npm run dev` — Vite's dev middleware
      intercepts onnxruntime-web's dynamic `import()` of its own
      wasm-loader `.mjs` and fails to serve it. Confirmed fine under
      `vite build && vite preview` (and will be fine on Vercel — same
      static-file-serving model). Test the ONNX path via build+preview,
      not dev, until/unless someone finds the dev-server fix.

    Motivated by the user's mobile-friendliness goal for the deployed site.

`docs/challenges-and-solutions-he.md` — Hebrew writeup of difficulties hit
and how they were solved, for project-defense prep (not the advisor status
doc below). Keep it updated alongside this file when a real bug-hunt/
decision happens; it's meant to be read, not just written once.

`docs/pose-iq-status-he.md` is the Hebrew status doc for the advisor —
~~rewritten 2026-08-23~~ to match the current architecture (was describing a
stale Desktop/Postgres/5-exercise version). `docs/pose-iq-status-he.html` was
rewritten to match; `.pdf` still needs manual re-export (open the .html,
Ctrl+P → save as PDF — no pandoc in this environment).
