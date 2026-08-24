import { useCallback, useEffect, useRef, useState } from 'react';
import { userApi } from '../api/client';
import type { AuthToken, ExerciseDef, RepPayload, User } from '../api/types';
import type { Tab } from '../components/NavBar';
import PoseFigure from '../components/PoseFigure';
import ScoreRing from '../components/ScoreRing';
import { useI18n } from '../i18n';
import { computeAngles } from '../pose/angles';
import { detectPose, loadPoseLandmarker } from '../pose/detector';
import { ERROR_TO_LINES, JOINT_ANCHOR, SKELETON_CONNECTIONS } from '../pose/landmarks';
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
  reps: number;
  liveForm: number;
  elapsed: number;
  issues: PostureIssue[];
  readiness: Record<string, ReadinessStatus>;
  tracking: boolean;
}

const EMPTY_HUD: Hud = {
  started: false, phase: '', instruction: '', phaseIndex: 0, phaseCount: 1,
  reps: 0, liveForm: 100, elapsed: 0, issues: [], readiness: {}, tracking: false,
};

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

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const smRef = useRef<ExerciseStateMachine | null>(null);
  const rulesRef = useRef<PostureRules | null>(null);
  const repErrorsRef = useRef<Set<string>>(new Set());
  const repsRef = useRef<RepPayload[]>([]);
  const startTimeRef = useRef(0);
  const endTimeRef = useRef(0);
  const lastDetectRef = useRef(0);
  const wasStartedRef = useRef(false);
  const flashTimerRef = useRef(0);
  const lastSpokenRef = useRef<Record<string, number>>({});
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

  const speak = useCallback((text: string) => {
    if (!voiceOnRef.current || !('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'he' ? 'he-IL' : 'en-US';
    u.rate = 1.1;
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

  /** Track every joint that's had a form issue this rep, and speak the
   * high-severity ones out loud — but not more than once per 4s per joint,
   * so a sustained issue doesn't spam the voice coach every frame. */
  function handlePostureIssues(issues: PostureIssue[], now: number) {
    for (const issue of issues) {
      repErrorsRef.current.add(issue.joint);
      if (issue.severity !== 'high') continue;
      const lastSpoken = lastSpokenRef.current[issue.joint] ?? 0;
      if (now - lastSpoken <= 4000) continue;
      lastSpokenRef.current[issue.joint] = now;
      speak(issue.message);
    }
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
      rulesRef.current = new PostureRules(
        user ? THRESHOLD_MODIFIER[user.fitness_level] : 1.0,
        user ? limitedJointsFor(user.limitations) : [],
      );
      repErrorsRef.current = new Set();
      repsRef.current = [];
      wasStartedRef.current = false;
      lastSpokenRef.current = {};
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

        let issues: PostureIssue[] = [];
        if (detection && result.started) {
          issues = rules.analyze(angles, sm.activeRules());
          handlePostureIssues(issues, now);

          if (!wasStartedRef.current) {
            wasStartedRef.current = true;
            showFlash(t.goFlash);
            speak(t.goFlash);
          }

          if (result.completedRep) {
            recordCompletedRep(result.repCount);
          }
        }

        drawOverlay(canvasRef.current, detection?.raw ?? null, vw, vh, issues,
                    result.started ? sm.currentPhase.diagnostic_joints : [], angles);

        setHud({
          started: result.started,
          phase: result.phase,
          instruction: result.instruction,
          phaseIndex: result.phaseIndex,
          phaseCount: result.phaseCount,
          reps: result.repCount,
          liveForm: repFormScore([...repErrorsRef.current]),
          elapsed: (now - startTimeRef.current) / 1000,
          issues,
          readiness: result.readiness,
          tracking: detection !== null,
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
  }, [user, speak, stopCamera, t]);

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
      <div className="max-w-2xl w-full mx-auto px-5 py-5 pb-16">
        <div className="relative rounded-xl overflow-hidden bg-black" style={{ minHeight: 240 }}>
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
    const weakJoints = [...new Set(repsRef.current.flatMap(r => r.error_joints))];
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
              {t.weakPrefix}: {weakJoints.map(j => t.jointLabels[j] ?? j).join(', ')}
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
