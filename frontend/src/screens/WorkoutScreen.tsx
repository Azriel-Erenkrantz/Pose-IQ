import { useCallback, useEffect, useRef, useState } from 'react';
import { userApi } from '../api/client';
import type { AuthToken, ExerciseDef, RepPayload, User } from '../api/types';
import type { Tab } from '../components/NavBar';
import PoseFigure from '../components/PoseFigure';
import ScoreRing from '../components/ScoreRing';
import { useI18n } from '../i18n';
import { computeAngles } from '../pose/angles';
import { DeltaTracker } from '../pose/deltaTracker';
import { detectPose, loadPoseLandmarker } from '../pose/detector';
import { ERROR_TO_LINES, JOINT_ANCHOR, SKELETON_CONNECTIONS } from '../pose/landmarks';
import { MlRepCounter } from '../pose/mlRepCounter';
import { classifyPhase, preloadPhaseModel } from '../pose/phaseClassifier';
import { PostureRules, repFormScore, THRESHOLD_MODIFIER, limitedJointsFor } from '../pose/postureRules';
import type { PostureIssue } from '../pose/postureRules';
import { ExerciseStateMachine } from '../pose/stateMachine';
import type { ReadinessStatus } from '../pose/stateMachine';

const DETECT_INTERVAL_MS = 33;   // ~30fps — the state machine's frame constants assume this

type Mode = 'setup' | 'live' | 'summary';

interface Hud {
  started: boolean;
  phase: string;
  instruction: string;
  phaseIndex: number;
  phaseCount: number;
  // Driven by the ML classifier (mlRepCounter.ts), not the rule engine —
  // see the live loop's comment on why, confirmed 2026-08-26.
  reps: number;
  liveForm: number;
  elapsed: number;
  issues: PostureIssue[];
  readiness: Record<string, ReadinessStatus>;
  tracking: boolean;
  // Per-tick ML phase call, shown so it can be eyeballed against the
  // rule-based phase (which still owns phase/instruction text above).
  mlDebug: { phase: string; confidence: number | null; agrees: boolean } | null;
}

const EMPTY_HUD: Hud = {
  started: false, phase: '', instruction: '', phaseIndex: 0, phaseCount: 1,
  reps: 0, liveForm: 100, elapsed: 0, issues: [], readiness: {}, tracking: false,
  mlDebug: null,
};

const ML_DEBUG_INTERVAL_MS = 250;   // model comparison doesn't need 30fps

interface Props {
  token: AuthToken;
  onNavigate: (tab: Tab) => void;
}

// ── Camera acquisition ───────────────────────────────────────────────────────

/** Try the ideal resolution first; some cameras reject those constraints, so
 * fall back to a bare video request before giving up. */
async function acquireCamera(): Promise<MediaStream> {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
  } catch (first: any) {
    try {
      return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (e: any) {
      throw new Error(e?.name ?? first?.name ?? 'UnknownError');
    }
  }
}

/** The <video> element mounts only after the mode switches to 'live', so a
 * single tick isn't always enough — poll briefly for the ref instead. */
async function waitForVideoElement(
  ref: React.RefObject<HTMLVideoElement | null>,
): Promise<HTMLVideoElement> {
  for (let i = 0; i < 40 && !ref.current; i++) {
    await new Promise(r => setTimeout(r, 25));
  }
  if (!ref.current) throw new Error('video element never mounted');
  return ref.current;
}

export default function WorkoutScreen({ token, onNavigate }: Props) {
  const { t, lang, exerciseName } = useI18n();

  const [mode, setMode] = useState<Mode>('setup');
  const [exercises, setExercises] = useState<ExerciseDef[] | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [selected, setSelected] = useState<ExerciseDef | null>(null);
  const [setupError, setSetupError] = useState('');
  const [starting, setStarting] = useState(false);
  const [hud, setHud] = useState<Hud>(EMPTY_HUD);
  const [flash, setFlash] = useState('');
  const [voiceOn, setVoiceOn] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [summaryWeight, setSummaryWeight] = useState('');
  const [targetReps, setTargetReps] = useState('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const smRef = useRef<ExerciseStateMachine | null>(null);
  const rulesRef = useRef<PostureRules | null>(null);
  const deltaTrackerRef = useRef<DeltaTracker | null>(null);
  const lastMlRunRef = useRef(0);
  const mlBusyRef = useRef(false);
  const mlDebugRef = useRef<Hud['mlDebug']>(null);
  const mlCounterRef = useRef<MlRepCounter | null>(null);
  const mlRepsRef = useRef(0);
  const targetRepsRef = useRef(0);
  const repErrorsRef = useRef<Set<string>>(new Set());
  const repsRef = useRef<RepPayload[]>([]);
  const startTimeRef = useRef(0);
  const endTimeRef = useRef(0);
  const lastDetectRef = useRef(0);
  const wasStartedRef = useRef(false);
  const flashTimerRef = useRef(0);
  const lastSpokenAtRef = useRef(0);
  const voiceOnRef = useRef(true);
  voiceOnRef.current = voiceOn;

  // ── Setup data ────────────────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([
      userApi.getExercises(),
      userApi.getUser(token.user_id, token.token),
    ])
      .then(([exs, u]) => { setExercises(exs); setUser(u); })
      .catch(() => setSetupError(t.couldNotLoad));
  }, [token, t]);

  // Start fetching the phase-classifier ONNX model as soon as the user picks
  // an exercise — in parallel with camera setup, not serially after it, so
  // it's (likely) already cached by the time the live loop needs it.
  useEffect(() => {
    if (selected) preloadPhaseModel(selected.id);
  }, [selected]);

  const speak = useCallback((text: string) => {
    if (!voiceOnRef.current || !('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'he' ? 'he-IL' : 'en-US';
    u.rate = 0.92;
    window.speechSynthesis.speak(u);
  }, [lang]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(tr => tr.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => () => {
    stopCamera();
    window.clearTimeout(flashTimerRef.current);
  }, [stopCamera]);

  function showFlash(text: string) {
    setFlash(text);
    window.clearTimeout(flashTimerRef.current);
    flashTimerRef.current = window.setTimeout(() => setFlash(''), 1200);
  }

  /** Track every joint that's had a form issue this rep, and speak *one*
   * high-severity issue at a time, at most once per 6s total (not per
   * joint) — several joints going out of frame at once (e.g. while the
   * user is still repositioning the camera) used to fire one utterance
   * per joint in the same tick, queuing up a rapid-fire burst. A single
   * global cooldown gives the user real time to react before the next
   * correction, and always speaks one coherent issue instead of several
   * queued ones stepping on each other. */
  function handlePostureIssues(issues: PostureIssue[], now: number) {
    for (const issue of issues) repErrorsRef.current.add(issue.joint);

    if (now - lastSpokenAtRef.current <= 6000) return;
    const top = issues.find(i => i.severity === 'high');
    if (!top) return;
    lastSpokenAtRef.current = now;
    speak(top.message);
  }

  /** Log a finished rep (with whichever joints had issues during it) and
   * reset the per-rep issue tracker for the next one. */
  function recordCompletedRep(repCount: number) {
    const errors = [...repErrorsRef.current];
    repsRef.current.push({
      rep_number: repCount,
      error_joints: errors,
      form_score: repFormScore(errors),
    });
    repErrorsRef.current = new Set();
    showFlash(t.repFlash(repCount));
    speak(String(repCount));
  }

  // ── Live loop ─────────────────────────────────────────────────────────────
  const startWorkout = useCallback(async (exercise: ExerciseDef) => {
    setStarting(true);
    setSetupError('');

    let landmarker;
    try {
      landmarker = await loadPoseLandmarker();
    } catch {
      setSetupError(t.modelError);
      setStarting(false);
      return;
    }

    let stream: MediaStream;
    try {
      stream = await acquireCamera();
    } catch (e: any) {
      setSetupError(`${t.cameraError} (${e.message})`);
      setStarting(false);
      return;
    }

    try {
      smRef.current = new ExerciseStateMachine(exercise);
      deltaTrackerRef.current = new DeltaTracker();
      mlDebugRef.current = null;
      mlCounterRef.current = new MlRepCounter(exercise);
      mlRepsRef.current = 0;
      targetRepsRef.current = Math.max(0, parseInt(targetReps, 10) || 0);
      rulesRef.current = new PostureRules(
        user ? THRESHOLD_MODIFIER[user.fitness_level] : 1.0,
        user ? limitedJointsFor(user.limitations) : [],
      );
      repErrorsRef.current = new Set();
      repsRef.current = [];
      wasStartedRef.current = false;
      lastSpokenAtRef.current = 0;
      startTimeRef.current = performance.now();
      setHud(EMPTY_HUD);
      setSaved(false);
      setSaveError('');
      setSummaryWeight('');
      setMode('live');

      streamRef.current = stream;
      const video = await waitForVideoElement(videoRef);
      video.srcObject = stream;
      await video.play();

      const loop = () => {
        rafRef.current = requestAnimationFrame(loop);
        const now = performance.now();
        if (now - lastDetectRef.current < DETECT_INTERVAL_MS) return;
        if (video.readyState < 2 || video.videoWidth === 0) return;
        lastDetectRef.current = now;

        const sm = smRef.current!;
        const rules = rulesRef.current!;
        const detection = detectPose(landmarker, video, now);
        const vw = video.videoWidth, vh = video.videoHeight;

        const angles = detection ? computeAngles(detection.named, vw, vh) : {};
        const result = sm.update(angles);

        // Rep counting/recording is driven by the ML classifier now, not the
        // rule engine — confirmed live 2026-08-26 across repeated 10-rep
        // sets: the rule engine's angle-range transitions kept getting
        // stuck at the "lowering"->"start" boundary (~40% of real reps
        // missed), while the ML majority-vote counter matched the true rep
        // count almost exactly. The rule engine (`sm`/`result`) still owns
        // the readiness/started gate below, but NOT posture-issue rules
        // (see the ML-phase lookup a few lines down) — a stuck `sm` phase
        // poisons those too: it flags real, correct motion as a form error
        // because it's comparing your actual (already-moved-on) angles
        // against the WRONG, stale phase's expected range. Confirmed live
        // 2026-08-26: a whole set of legitimate reps ended with "weak
        // points: both shoulders, both elbows, spine" — every joint that
        // moves, because the reference phase never caught up.
        if (detection && result.started && now - lastMlRunRef.current >= ML_DEBUG_INTERVAL_MS && !mlBusyRef.current) {
          lastMlRunRef.current = now;
          mlBusyRef.current = true;
          const deltas = deltaTrackerRef.current!.update(angles, now);
          const sourcePhase = result.phase;
          classifyPhase(exercise.id, angles, deltas)
            .then(pred => {
              if (pred) {
                mlDebugRef.current = { phase: pred.phase, confidence: pred.confidence, agrees: pred.phase === sourcePhase };
                const maxMotion = Object.values(deltas).reduce((m, v) => Math.max(m, Math.abs(v)), 0);
                const mlResult = mlCounterRef.current!.update(pred.phase, pred.confidence, now, maxMotion);
                mlRepsRef.current = mlResult.repCount;
                if (mlResult.completedRep) {
                  recordCompletedRep(mlResult.repCount);
                  // Auto-end the instant the target is hit — confirmed live
                  // 2026-08-26: leaving the camera running while the user
                  // then moves toward the device to close the workout
                  // manually risks that motion itself getting classified as
                  // one more rep (no "idle" class exists — see
                  // mlRepCounter.ts). Ending immediately removes that window.
                  if (targetRepsRef.current > 0 && mlResult.repCount >= targetRepsRef.current) {
                    endWorkout();
                  }
                }
              }
            })
            .catch(err => console.error('[phaseClassifier] inference failed:', err))
            .finally(() => { mlBusyRef.current = false; });
        }

        // Which phase's angle ranges to hold the user to right now — prefer
        // the ML's own last raw phase call over `sm.currentPhase` for this,
        // since `sm`'s phase pointer is the thing that gets stuck (see the
        // comment above). Falls back to `sm` only before the first ML tick
        // has run (e.g. the very start of the set).
        const posturePhase = mlDebugRef.current
          ? exercise.phases.find(p => p.name === mlDebugRef.current!.phase)
          : null;
        const activeRules = posturePhase
          ? { ...exercise.global_constraints, ...posturePhase.angles }
          : sm.activeRules();

        let issues: PostureIssue[] = [];
        if (detection && result.started) {
          issues = rules.analyze(angles, activeRules);
          handlePostureIssues(issues, now);

          if (!wasStartedRef.current) {
            wasStartedRef.current = true;
            showFlash(t.goFlash);
            speak(t.goFlash);
          }
        }

        drawOverlay(canvasRef.current, detection?.raw ?? null, vw, vh, issues,
                    result.started ? (posturePhase ?? sm.currentPhase).diagnostic_joints : [], angles);

        setHud({
          started: result.started,
          phase: result.phase,
          instruction: result.instruction,
          phaseIndex: result.phaseIndex,
          phaseCount: result.phaseCount,
          reps: mlRepsRef.current,
          liveForm: repFormScore([...repErrorsRef.current]),
          elapsed: (now - startTimeRef.current) / 1000,
          issues,
          readiness: result.readiness,
          tracking: detection !== null,
          mlDebug: mlDebugRef.current,
        });
      };
      rafRef.current = requestAnimationFrame(loop);
    } catch (e: any) {
      console.error('startWorkout failed after camera acquisition:', e);
      stopCamera();
      setMode('setup');
      setSetupError(`${t.cameraError} [${e?.name ?? 'Error'}: ${e?.message ?? ''}]`);
    } finally {
      setStarting(false);
    }
  }, [user, speak, stopCamera, t, targetReps]);

  function endWorkout() {
    endTimeRef.current = performance.now();
    stopCamera();
    window.speechSynthesis?.cancel();
    setMode(repsRef.current.length > 0 ? 'summary' : 'setup');
  }

  async function saveSession() {
    if (!selected) return;
    setSaving(true);
    setSaveError('');
    try {
      const weight = parseFloat(summaryWeight);
      await userApi.saveWorkoutSession(token.user_id, {
        exercise_id: selected.id,
        exercise_name: selected.name,
        duration_seconds: Math.round((endTimeRef.current - startTimeRef.current) / 100) / 10,
        weight_kg: isNaN(weight) ? null : weight,
        reps: repsRef.current,
      }, token.token);
      setSaved(true);
      setTimeout(() => onNavigate('history'), 1200);
    } catch (e: any) {
      setSaveError(e.message ?? t.couldNotLoad);
    } finally {
      setSaving(false);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  if (mode === 'live') {
    const mins = Math.floor(hud.elapsed / 60), secs = Math.floor(hud.elapsed % 60);
    return (
      <div className="max-w-4xl w-full mx-auto px-5 py-5 pb-16">
        {/* Bigger than the app's usual max-w-2xl column — this screen's whole
            point is seeing yourself clearly enough to check full-body framing. */}
        <div className="relative rounded-xl overflow-hidden bg-black" style={{ minHeight: 320 }}>
          <video ref={videoRef} playsInline muted
                 className="block w-full" style={{ transform: 'scaleX(-1)' }} />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

          {/* Top HUD */}
          <div className="absolute top-0 inset-x-0 flex items-start justify-between p-3">
            <div className="bg-black/60 rounded-lg px-3 py-1.5">
              <p className="text-white font-bold text-sm leading-tight">
                {selected ? exerciseName(selected.id, selected.name) : ''}
              </p>
              <p className={`text-xs mt-0.5 ${hud.started ? 'text-[#7ee2a8]' : 'text-[#ffd37a]'}`}>
                {hud.started
                  ? `${hud.phase} (${hud.phaseIndex + 1}/${hud.phaseCount})${hud.instruction ? ' — ' + hud.instruction : ''}`
                  : t.getIntoPosition}
              </p>
              {/* Leftover from validating the ONNX classifier against the rule
                  engine live (see phaseClassifier.ts) — shown to every user
                  unconditionally, not gated behind a dev flag. Harmless (it's
                  informational, not interactive) but was meant as a temporary
                  comparison view, not permanent UI; worth hiding or removing
                  now that mlRepCounter.ts is the real decision-maker and
                  there's nothing left to compare it against for the user. */}
              {hud.mlDebug && (
                <p className={`text-[10px] mt-0.5 ${hud.mlDebug.agrees ? 'text-[#7ee2a8]/70' : 'text-[#ff9a7a]'}`}>
                  ML: {hud.mlDebug.phase}
                  {hud.mlDebug.confidence !== null ? ` (${Math.round(hud.mlDebug.confidence * 100)}%)` : ''}
                  {hud.mlDebug.agrees ? ' ✓' : ' ≠ rules'}
                </p>
              )}
            </div>
            <div className="bg-black/60 rounded-lg px-3 py-1.5 text-end" dir="ltr">
              <p className="text-white font-black text-lg num leading-tight">
                {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
              </p>
              <p className="text-[#bbb] text-xs num">{t.repsWord}: {hud.reps}</p>
            </div>
          </div>

          {/* Center flash */}
          {flash && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <span className="text-white font-black text-5xl drop-shadow-[0_2px_12px_rgba(0,0,0,.8)]">
                {flash}
              </span>
            </div>
          )}

          {/* Readiness checklist (before start) */}
          {!hud.started && Object.keys(hud.readiness).length > 0 && (
            <div className="absolute bottom-3 start-3 bg-black/60 rounded-lg px-3 py-2 max-w-[75%]">
              <p className="text-[#bbb] text-xs font-bold mb-1">
                {t.startingPosition}{' '}
                <span className="num">
                  {Object.values(hud.readiness).filter(s => s === null).length}/{Object.keys(hud.readiness).length}
                </span>
              </p>
              {Object.entries(hud.readiness).map(([joint, status]) => (
                <p key={joint} className="text-xs leading-relaxed">
                  <span className={status === null ? 'text-[#7ee2a8]' : 'text-[#ffd37a]'}>
                    {status === null ? '●' : '○'}
                  </span>{' '}
                  <span className="text-white">{t.jointLabels[joint] ?? joint}</span>
                  {status !== null && (
                    <span className="text-[#ffd37a]">
                      {' — '}
                      {status === 'missing' ? t.adjustMissing
                        : status === 'too low' ? t.adjustStraighten : t.adjustBend}
                    </span>
                  )}
                </p>
              ))}
            </div>
          )}

          {/* Feedback banner (during exercise) */}
          {hud.started && (
            <div className="absolute bottom-3 inset-x-3">
              {hud.issues.length > 0 ? (
                <div className="bg-[#b42318]/85 rounded-lg px-3 py-2">
                  {hud.issues.slice(0, 2).map(issue => (
                    <p key={issue.joint} className="text-white text-sm font-bold leading-relaxed">
                      {issue.message}
                    </p>
                  ))}
                </div>
              ) : hud.tracking ? (
                <div className="bg-black/50 rounded-lg px-3 py-1.5 inline-block">
                  <p className="text-[#7ee2a8] text-xs font-bold">{t.goodForm}</p>
                </div>
              ) : null}
            </div>
          )}
        </div>

        {/* Stat strip */}
        <div className="panel flex items-center mt-4">
          <div className="flex-1 px-4 py-3">
            <p className="text-[#171716] font-black text-3xl num leading-none">{hud.reps}</p>
            <p className="label mt-1.5">{t.repsWord}</p>
          </div>
          <div className="w-px bg-[#e6e5e1] my-3" />
          <div className="flex-1 px-4 py-3 flex items-center gap-3">
            <ScoreRing score={hud.liveForm} size={44} />
            <p className="label">{t.formLabel}</p>
          </div>
          <div className="w-px bg-[#e6e5e1] my-3" />
          <div className="flex-1 px-4 py-3" dir="ltr">
            <p className="text-[#171716] font-black text-3xl num leading-none">
              {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
            </p>
            <p className="label mt-1.5">{t.timeLabel}</p>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-4">
          <button onClick={endWorkout} className="btn flex-1 text-base py-3">
            {t.endWorkout}
          </button>
          <button
            onClick={() => setVoiceOn(v => !v)}
            className={`btn-ghost text-xs font-bold px-3 py-3 ${voiceOn ? 'text-accent' : ''}`}
          >
            {t.voiceLabel} {voiceOn ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>
    );
  }

  if (mode === 'summary') {
    const scores = repsRef.current.map(r => r.form_score);
    const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

    // A joint counts as a real "weak point" only if it recurs across a real
    // share of the reps, not just once — a single flagged rep (tracking
    // blip, one awkward moment) isn't a pattern. Requires both a minimum
    // count and a minimum fraction so short sets don't flag on one rep, but
    // longer sets still catch something that's only occasionally off.
    const jointRepCounts = new Map<string, number>();
    for (const r of repsRef.current) {
      for (const joint of r.error_joints) {
        jointRepCounts.set(joint, (jointRepCounts.get(joint) ?? 0) + 1);
      }
    }
    const weakThreshold = Math.max(2, Math.ceil(repsRef.current.length * 0.3));
    const weakJoints = [...jointRepCounts.entries()]
      .filter(([, count]) => count >= weakThreshold)
      .sort((a, b) => b[1] - a[1]);

    const duration = (endTimeRef.current - startTimeRef.current) / 1000;
    const mins = Math.floor(duration / 60), secs = Math.floor(duration % 60);

    return (
      <div className="max-w-2xl w-full mx-auto px-5 py-7 pb-16">
        <h1 className="text-4xl font-black text-[#171716] mb-7">{t.summaryTitle}</h1>

        <div className="panel p-5 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-[#171716] font-bold text-lg">
                {selected ? exerciseName(selected.id, selected.name) : ''}
              </p>
              <p className="text-[#a09f98] text-sm num mt-0.5" dir="ltr">
                {repsRef.current.length} {t.repsWord} · {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
              </p>
            </div>
            <ScoreRing score={avg} size={64} />
          </div>

          {weakJoints.length > 0 && (
            <p className="text-[#b45309] text-sm mb-4">
              {t.weakPrefix}: {weakJoints
                .map(([joint, count]) => `${t.jointLabels[joint] ?? joint} (${count}/${repsRef.current.length})`)
                .join(', ')}
            </p>
          )}

          <label className="label block mb-1.5">{t.weightUsedLabel}</label>
          <input
            type="number" min={0} max={300} step={0.5} dir="ltr"
            className="field max-w-[160px]"
            value={summaryWeight}
            onChange={e => setSummaryWeight(e.target.value)}
          />
        </div>

        {saveError && <p className="text-[#d92d20] text-sm mb-3">{saveError}</p>}

        <div className="flex items-center gap-3">
          <button onClick={saveSession} disabled={saving || saved}
                  className="btn flex-1 text-base py-3.5 flex items-center justify-center gap-2">
            {saving ? <><span className="spinner-sm" /> {t.savingSession}</>
              : saved ? t.sessionSaved : t.saveSession}
          </button>
          {!saved && (
            <button onClick={() => setMode('setup')} className="btn-ghost text-sm px-5 py-3.5">
              {t.discard}
            </button>
          )}
        </div>
      </div>
    );
  }

  // ── Setup ──
  return (
    <div className="max-w-2xl w-full mx-auto px-5 py-7 pb-16">
      <h1 className="text-4xl font-black text-[#171716] mb-3 leading-tight">{t.workoutTitle}</h1>
      <p className="text-[#6f6e68] text-sm max-w-md leading-relaxed mb-8">{t.workoutPitch}</p>

      <p className="label mb-2.5">{t.chooseExercise}</p>
      {setupError && <p className="text-[#d92d20] text-sm mb-3">{setupError}</p>}
      {exercises === null ? (
        <div className="panel p-6 flex justify-center"><div className="spinner" /></div>
      ) : (
        <div className="panel rows mb-5">
          {exercises.map(ex => (
            <button
              key={ex.id}
              disabled={!ex.ready}
              onClick={() => setSelected(ex)}
              className="w-full flex items-center justify-between px-4 py-3.5 gap-3 text-start disabled:opacity-45"
            >
              <div className="flex items-center gap-3">
                <span className={`w-2.5 h-2.5 rounded-[2px] flex-shrink-0 ${
                  selected?.id === ex.id ? 'bg-[#0e7a4a]' : 'border border-[#cfceca]'
                }`} />
                <div>
                  <p className="text-[#171716] font-bold text-[15px] leading-tight">
                    {exerciseName(ex.id, ex.name)}
                  </p>
                  {!ex.ready && (
                    <p className="text-[#b45309] text-xs mt-0.5">{t.notReadyHint}</p>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      <p className="text-[#a09f98] text-xs mb-4">{t.cameraHint}</p>

      <label className="label block mb-1.5">{t.targetRepsLabel}</label>
      <input
        type="number" min={0} max={50} step={1} dir="ltr"
        className="field max-w-[120px] mb-6"
        value={targetReps}
        onChange={e => setTargetReps(e.target.value)}
      />

      <button
        onClick={() => selected && startWorkout(selected)}
        disabled={!selected || starting}
        className="btn w-full text-base py-3.5 flex items-center justify-center gap-2 mb-8"
      >
        {starting ? <><span className="spinner-sm" /> {t.loadingModel}</> : t.startWorkout}
      </button>

      <div className="panel p-6">
        <p className="label mb-4">{t.formTracking}</p>
        <PoseFigure />
      </div>
    </div>
  );
}

// ── Canvas overlay ────────────────────────────────────────────────────────────
// Skeleton + joints + angle chips, drawn mirrored to match the selfie video.
// Split into one function per visual layer — each just needs the raw
// landmarks and a couple of mirror-coordinate helpers.

type RawPoint = { x: number; y: number; visibility: number };
type MirrorFn = (n: number) => number;

const MIN_DRAW_VISIBILITY = 0.4;

/** Which skeleton lines and joints should render red — the ones a current
 * posture issue is about. */
function issueHighlights(issues: PostureIssue[]): { redLines: Set<string>; badJoints: Set<string> } {
  const redLines = new Set<string>();
  const badJoints = new Set<string>();
  for (const issue of issues) {
    badJoints.add(issue.joint);
    for (const [s, e] of ERROR_TO_LINES[issue.joint] ?? []) {
      redLines.add(`${s}-${e}`);
    }
  }
  return { redLines, badJoints };
}

function drawSkeleton(ctx: CanvasRenderingContext2D, raw: RawPoint[], redLines: Set<string>,
                      mx: MirrorFn, my: MirrorFn, vw: number) {
  ctx.lineWidth = Math.max(3, vw / 320);
  ctx.lineCap = 'round';
  for (const [s, e] of SKELETON_CONNECTIONS) {
    const ps = raw[s], pe = raw[e];
    if (!ps || !pe || ps.visibility < MIN_DRAW_VISIBILITY || pe.visibility < MIN_DRAW_VISIBILITY) continue;
    ctx.strokeStyle = redLines.has(`${s}-${e}`) ? '#ff5c5c' : 'rgba(255,255,255,.92)';
    ctx.beginPath();
    ctx.moveTo(mx(ps.x), my(ps.y));
    ctx.lineTo(mx(pe.x), my(pe.y));
    ctx.stroke();
  }
}

/** Draws a dot per visible joint (excluding face landmarks 0-10) and
 * returns the dot radius used, so the angle chips can size themselves
 * relative to it. */
function drawJoints(ctx: CanvasRenderingContext2D, raw: RawPoint[], badJoints: Set<string>,
                    mx: MirrorFn, my: MirrorFn, vw: number): number {
  const badAnchors = new Set<number>();
  for (const j of badJoints) {
    if (JOINT_ANCHOR[j] !== undefined) badAnchors.add(JOINT_ANCHOR[j]);
  }
  const r = Math.max(5, vw / 220);
  raw.forEach((p, idx) => {
    if (idx <= 10 || p.visibility < MIN_DRAW_VISIBILITY) return;
    ctx.fillStyle = badAnchors.has(idx) ? '#ff5c5c' : '#2fbf71';
    ctx.beginPath();
    ctx.arc(mx(p.x), my(p.y), badAnchors.has(idx) ? r * 1.4 : r, 0, Math.PI * 2);
    ctx.fill();
  });
  return r;
}

/** Labeled angle badges next to the current phase's diagnostic joints. */
function drawAngleChips(ctx: CanvasRenderingContext2D, raw: RawPoint[], diagnosticJoints: string[],
                        angles: Record<string, number>, mx: MirrorFn, my: MirrorFn, vw: number, jointRadius: number) {
  const chipFont = Math.max(13, vw / 55);
  ctx.font = `800 ${chipFont}px Heebo, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const joint of diagnosticJoints) {
    const idx = JOINT_ANCHOR[joint];
    const angle = angles[joint];
    if (idx === undefined || angle === undefined) continue;
    const p = raw[idx];
    if (!p || p.visibility < MIN_DRAW_VISIBILITY) continue;

    const label = `${Math.round(angle)}°`;
    const w = ctx.measureText(label).width + chipFont;
    const h = chipFont * 1.7;
    const cx = mx(p.x) + jointRadius * 3 + w / 2;
    const cy = my(p.y) - h;

    ctx.fillStyle = 'rgba(23,23,22,.9)';
    ctx.beginPath();
    ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 6);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(label, cx, cy + 1);
  }
}

function drawOverlay(
  canvas: HTMLCanvasElement | null,
  raw: RawPoint[] | null,
  vw: number,
  vh: number,
  issues: PostureIssue[],
  diagnosticJoints: string[],
  angles: Record<string, number>,
) {
  if (!canvas) return;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, vw, vh);
  if (!raw) return;

  const mx: MirrorFn = (x) => (1 - x) * vw;   // mirror to match scaleX(-1) video
  const my: MirrorFn = (y) => y * vh;

  const { redLines, badJoints } = issueHighlights(issues);
  drawSkeleton(ctx, raw, redLines, mx, my, vw);
  const jointRadius = drawJoints(ctx, raw, badJoints, mx, my, vw);
  drawAngleChips(ctx, raw, diagnosticJoints, angles, mx, my, vw, jointRadius);
}
