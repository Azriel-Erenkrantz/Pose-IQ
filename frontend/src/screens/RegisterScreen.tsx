import React, { useEffect, useState } from 'react';
import { authApi } from '../api/client';
import type {
  AuthToken,
  Equipment,
  FitnessLevel,
  ProfileSetupOptions,
  TargetGoal,
  TrainerPersonality,
} from '../api/types';

interface Props {
  onRegister: (token: AuthToken) => void;
  onGoToLogin: () => void;
}

const TOTAL_STEPS = 6;

const STEP_TITLES = [
  'Create your account',
  'Fitness level',
  'Training goals',
  'Equipment',
  'Coach style',
  'Any limitations?',
];

const FITNESS_LABELS: Record<FitnessLevel, string> = {
  beginner:     'Beginner',
  intermediate: 'Intermediate',
  advanced:     'Advanced',
};

const FITNESS_DESC: Record<FitnessLevel, string> = {
  beginner:     'New to fitness or returning after a break',
  intermediate: 'Exercising regularly for 6+ months',
  advanced:     'Training consistently for 2+ years',
};

const GOAL_LABELS: Record<TargetGoal, string> = {
  legs:      '🦵 Legs',
  cardio:    '❤️ Cardio',
  abs:       '💪 Abs',
  arms:      '🤜 Arms',
  full_body: '🏋️ Full Body',
  other:     '✨ Other',
};

const EQUIPMENT_LABELS: Record<Equipment, string> = {
  dumbbells:        '🏋️ Dumbbells',
  resistance_bands: '🔁 Bands',
  none:             '🙌 No Equipment',
};

const TRAINER_LABELS: Record<TrainerPersonality, string> = {
  tough:      '🔥 Tough',
  calm:       '🧘 Calm',
  motivating: '⚡ Motivating',
};

const TRAINER_DESC: Record<TrainerPersonality, string> = {
  tough:      'Direct and demanding — short commands, no fluff',
  calm:       'Gentle and patient — smooth, reassuring cues',
  motivating: 'Energetic and encouraging — hypes you up',
};

const LIMITATION_LABELS: Record<string, string> = {
  right_knee:     '🦵 Right Knee',
  left_knee:      '🦵 Left Knee',
  lower_back:     '🔙 Lower Back',
  right_shoulder: '💪 Right Shoulder',
  left_shoulder:  '💪 Left Shoulder',
  right_elbow:    '🤜 Right Elbow',
  left_elbow:     '🤛 Left Elbow',
};

function toggleMulti<T>(list: T[], item: T): T[] {
  return list.includes(item) ? list.filter(i => i !== item) : [...list, item];
}

export default function RegisterScreen({ onRegister, onGoToLogin }: Props) {
  const [step, setStep]       = useState(1);
  const [options, setOptions] = useState<ProfileSetupOptions | null>(null);

  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [fitnessLevel, setFitnessLevel] = useState<FitnessLevel>('intermediate');
  const [goals, setGoals]               = useState<TargetGoal[]>([]);
  const [equipment, setEquipment]       = useState<Equipment[]>([]);
  const [trainer, setTrainer]           = useState<TrainerPersonality>('motivating');
  const [limitations, setLimitations]   = useState<string[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  useEffect(() => { authApi.options().then(setOptions).catch(() => {}); }, []);

  function validateStep(): string {
    if (step === 1) {
      if (!name.trim())  return 'Please enter your name.';
      if (!email.trim()) return 'Please enter your email.';
      if (!/\S+@\S+\.\S+/.test(email)) return 'Please enter a valid email.';
      if (password.length < 6) return 'Password must be at least 6 characters.';
    }
    if (step === 3 && goals.length === 0)     return 'Pick at least one goal.';
    if (step === 4 && equipment.length === 0) return 'Pick at least one equipment option.';
    return '';
  }

  function handleNext() {
    const e = validateStep();
    if (e) { setError(e); return; }
    setError('');
    setStep(s => s + 1);
  }

  async function handleSubmit() {
    setLoading(true);
    setError('');
    try {
      const token = await authApi.register({
        name:                name.trim(),
        email:               email.trim().toLowerCase(),
        password,
        fitness_level:       fitnessLevel,
        trainer_personality: trainer,
        target_goals:        goals,
        equipment,
        limitations,
      });
      onRegister(token);
    } catch (err: any) {
      setError(err.message ?? 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  const progressPct = ((step - 1) / (TOTAL_STEPS - 1)) * 100;

  return (
    <div className="min-h-screen bg-[#0f0f1a] flex items-center justify-center px-6 py-10">
      <div className="w-full max-w-md flex flex-col gap-6">
        {/* Progress bar */}
        <div>
          <div className="flex justify-between text-xs text-[#666] mb-2">
            <span>Step {step} of {TOTAL_STEPS}</span>
            <span>{STEP_TITLES[step - 1]}</span>
          </div>
          <div className="h-1 bg-[#2a2a40] rounded-full overflow-hidden">
            <div
              className="h-full bg-[#4ade80] rounded-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white">{STEP_TITLES[step - 1]}</h2>

        {/* Step content */}
        <div className="flex flex-col gap-3">
          {/* Step 1 — Account */}
          {step === 1 && (
            <>
              <Field label="Name">
                <input
                  className="field-input w-full bg-[#1a1a2e] text-white border border-[#2a2a40] rounded-xl px-4 py-3 focus:outline-none focus:border-[#4ade80]"
                  placeholder="Your name"
                  value={name}
                  onChange={e => setName(e.target.value)}
                />
              </Field>
              <Field label="Email">
                <input
                  type="email"
                  className="w-full bg-[#1a1a2e] text-white border border-[#2a2a40] rounded-xl px-4 py-3 focus:outline-none focus:border-[#4ade80]"
                  placeholder="you@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </Field>
              <Field label="Password">
                <input
                  type="password"
                  className="w-full bg-[#1a1a2e] text-white border border-[#2a2a40] rounded-xl px-4 py-3 focus:outline-none focus:border-[#4ade80]"
                  placeholder="At least 6 characters"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                />
              </Field>
            </>
          )}

          {/* Step 2 — Fitness level */}
          {step === 2 && (['beginner', 'intermediate', 'advanced'] as FitnessLevel[]).map(level => (
            <button
              key={level}
              type="button"
              onClick={() => setFitnessLevel(level)}
              className={`text-left p-4 rounded-xl border transition-colors ${
                fitnessLevel === level
                  ? 'border-[#4ade80] bg-[#1a2e1a]'
                  : 'border-[#2a2a40] bg-[#1a1a2e] hover:border-[#3a3a50]'
              }`}
            >
              <div className="flex items-center gap-3 mb-1">
                <span className={`w-4 h-4 rounded-full border-2 flex-shrink-0 ${fitnessLevel === level ? 'border-[#4ade80] bg-[#4ade80]' : 'border-[#444]'}`} />
                <span className="text-white font-semibold">{FITNESS_LABELS[level]}</span>
              </div>
              <p className="text-[#888] text-sm ml-7">{FITNESS_DESC[level]}</p>
            </button>
          ))}

          {/* Step 3 — Goals */}
          {step === 3 && (
            <div className="grid grid-cols-3 gap-3">
              {(options?.target_goals ?? Object.keys(GOAL_LABELS) as TargetGoal[]).map(goal => {
                const [icon, ...words] = GOAL_LABELS[goal]?.split(' ') ?? [goal];
                return (
                  <button
                    key={goal}
                    type="button"
                    onClick={() => setGoals(g => toggleMulti(g, goal))}
                    className={`flex flex-col items-center justify-center gap-1 p-4 rounded-xl border min-h-[80px] transition-colors ${
                      goals.includes(goal)
                        ? 'border-[#4ade80] bg-[#1a2e1a]'
                        : 'border-[#2a2a40] bg-[#1a1a2e] hover:border-[#3a3a50]'
                    }`}
                  >
                    <span className="text-2xl">{icon}</span>
                    <span className="text-[#ccc] text-xs font-medium text-center">{words.join(' ') || goal}</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Step 4 — Equipment */}
          {step === 4 && (
            <div className="grid grid-cols-3 gap-3">
              {(options?.equipment_options ?? Object.keys(EQUIPMENT_LABELS) as Equipment[]).map(eq => {
                const [icon, ...words] = EQUIPMENT_LABELS[eq]?.split(' ') ?? [eq];
                return (
                  <button
                    key={eq}
                    type="button"
                    onClick={() => setEquipment(e => toggleMulti(e, eq))}
                    className={`flex flex-col items-center justify-center gap-1 p-4 rounded-xl border min-h-[80px] transition-colors ${
                      equipment.includes(eq)
                        ? 'border-[#4ade80] bg-[#1a2e1a]'
                        : 'border-[#2a2a40] bg-[#1a1a2e] hover:border-[#3a3a50]'
                    }`}
                  >
                    <span className="text-2xl">{icon}</span>
                    <span className="text-[#ccc] text-xs font-medium text-center">{words.join(' ') || eq}</span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Step 5 — Trainer */}
          {step === 5 && (['tough', 'calm', 'motivating'] as TrainerPersonality[]).map(style => (
            <button
              key={style}
              type="button"
              onClick={() => setTrainer(style)}
              className={`text-left p-4 rounded-xl border transition-colors ${
                trainer === style
                  ? 'border-[#4ade80] bg-[#1a2e1a]'
                  : 'border-[#2a2a40] bg-[#1a1a2e] hover:border-[#3a3a50]'
              }`}
            >
              <div className="flex items-center gap-3 mb-1">
                <span className={`w-4 h-4 rounded-full border-2 flex-shrink-0 ${trainer === style ? 'border-[#4ade80] bg-[#4ade80]' : 'border-[#444]'}`} />
                <span className="text-white font-semibold">{TRAINER_LABELS[style]}</span>
              </div>
              <p className="text-[#888] text-sm ml-7">{TRAINER_DESC[style]}</p>
            </button>
          ))}

          {/* Step 6 — Limitations */}
          {step === 6 && (
            <>
              <p className="text-[#888] text-sm">
                Select any joints that need extra care. The app will adapt form checks accordingly. Skip if none apply.
              </p>
              <div className="grid grid-cols-3 gap-3">
                {(options?.limitation_options ?? Object.keys(LIMITATION_LABELS)).map(lim => {
                  const [icon, ...words] = (LIMITATION_LABELS[lim] ?? lim).split(' ');
                  return (
                    <button
                      key={lim}
                      type="button"
                      onClick={() => setLimitations(l => toggleMulti(l, lim))}
                      className={`flex flex-col items-center justify-center gap-1 p-4 rounded-xl border min-h-[80px] transition-colors ${
                        limitations.includes(lim)
                          ? 'border-[#4ade80] bg-[#1a2e1a]'
                          : 'border-[#2a2a40] bg-[#1a1a2e] hover:border-[#3a3a50]'
                      }`}
                    >
                      <span className="text-2xl">{icon}</span>
                      <span className="text-[#ccc] text-xs font-medium text-center">{words.join(' ') || lim}</span>
                    </button>
                  );
                })}
              </div>
            </>
          )}

          {error && <p className="text-[#ff6b6b] text-sm text-center">{error}</p>}
        </div>

        {/* Nav buttons */}
        <div className="flex gap-3 items-center">
          <button
            type="button"
            onClick={step > 1 ? () => { setError(''); setStep(s => s - 1); } : onGoToLogin}
            className="text-[#888] text-sm px-4 py-3 hover:text-white transition-colors"
          >
            {step > 1 ? '← Back' : 'Log in'}
          </button>

          {step < TOTAL_STEPS ? (
            <button
              type="button"
              onClick={handleNext}
              className="flex-1 bg-[#4ade80] text-[#0f0f1a] font-bold text-base rounded-xl py-4 hover:bg-[#22c55e] transition-colors"
            >
              Next →
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className="flex-1 bg-[#4ade80] text-[#0f0f1a] font-bold text-base rounded-xl py-4 hover:bg-[#22c55e] transition-colors disabled:opacity-60 flex items-center justify-center"
            >
              {loading ? <span className="spinner-sm" /> : 'Create Account'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[#aaa] text-xs font-semibold mb-1 tracking-wide uppercase">{label}</label>
      {children}
    </div>
  );
}
