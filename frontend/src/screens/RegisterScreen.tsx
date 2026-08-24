import React, { useEffect, useState } from 'react';
import { authApi } from '../api/client';
import type {
  AuthToken,
  FitnessLevel,
  ProfileSetupOptions,
} from '../api/types';
import { useI18n } from '../i18n';

interface Props {
  onRegister: (token: AuthToken) => void;
  onGoToLogin: () => void;
}

const TOTAL_STEPS = 3;

function toggleMulti<T>(list: T[], item: T): T[] {
  return list.includes(item) ? list.filter(i => i !== item) : [...list, item];
}

export default function RegisterScreen({ onRegister, onGoToLogin }: Props) {
  const { t } = useI18n();
  const [step, setStep]       = useState(1);
  const [options, setOptions] = useState<ProfileSetupOptions | null>(null);

  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [fitnessLevel, setFitnessLevel] = useState<FitnessLevel>('intermediate');
  const [limitations, setLimitations]   = useState<string[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  useEffect(() => { authApi.options().then(setOptions).catch(() => {}); }, []);

  function validateStep(): string {
    if (step === 1) {
      if (!name.trim())  return t.errName;
      if (!email.trim()) return t.errEmail;
      if (!/\S+@\S+\.\S+/.test(email)) return t.errEmailValid;
      if (password.length < 6) return t.errPassword;
    }
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
        name:          name.trim(),
        email:         email.trim().toLowerCase(),
        password,
        fitness_level: fitnessLevel,
        limitations,
      });
      onRegister(token);
    } catch (err: any) {
      setError(err.message ?? t.registerFailed);
    } finally {
      setLoading(false);
    }
  }

  const progressPct = ((step - 1) / (TOTAL_STEPS - 1)) * 100;

  return (
    <div className="min-h-screen flex items-center justify-center px-6 py-10">
      <div className="w-full max-w-md flex flex-col gap-6">
        {/* Progress bar */}
        <div>
          <div className="flex justify-between mb-2">
            <span className="label num">{t.stepOf(step, TOTAL_STEPS)}</span>
            <span className="label">{t.stepTitles[step - 1]}</span>
          </div>
          <div className="h-[3px] bg-[#e6e5e1] rounded-full overflow-hidden">
            <div
              className="h-full bg-[#171716] transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        <h2 className="text-3xl font-black text-[#171716]">{t.stepTitles[step - 1]}</h2>

        {/* Step content */}
        <div className="flex flex-col gap-3">
          {/* Step 1 — Account */}
          {step === 1 && (
            <>
              <Field label={t.nameLabel}>
                <input
                  className="field"
                  placeholder={t.namePh}
                  value={name}
                  onChange={e => setName(e.target.value)}
                />
              </Field>
              <Field label={t.email}>
                <input
                  type="email"
                  className="field"
                  placeholder="you@example.com"
                  dir="ltr"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </Field>
              <Field label={t.password}>
                <input
                  type="password"
                  className="field"
                  placeholder={t.passwordPh}
                  dir="ltr"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                />
              </Field>
            </>
          )}

          {/* Step 2 — Fitness level */}
          {step === 2 && (['beginner', 'intermediate', 'advanced'] as FitnessLevel[]).map(level => (
            <RadioTile
              key={level}
              selected={fitnessLevel === level}
              onClick={() => setFitnessLevel(level)}
              title={t.fitnessLabels[level]}
              desc={t.fitnessDesc[level]}
            />
          ))}

          {/* Step 3 — Limitations */}
          {step === 3 && (
            <>
              <p className="text-[#8b87a0] text-sm">{t.limitationsHint}</p>
              <div className="grid grid-cols-3 gap-3">
                {(options?.limitation_options ?? Object.keys(t.limitationLabels)).map(lim => (
                  <IconTile
                    key={lim}
                    selected={limitations.includes(lim)}
                    onClick={() => setLimitations(l => toggleMulti(l, lim))}
                    label={t.limitationLabels[lim] ?? lim}
                  />
                ))}
              </div>
            </>
          )}

          {error && <p className="text-[#d92d20] text-sm text-center">{error}</p>}
        </div>

        {/* Nav buttons */}
        <div className="flex gap-3 items-center">
          <button
            type="button"
            onClick={step > 1 ? () => { setError(''); setStep(s => s - 1); } : onGoToLogin}
            className="text-[#6f6e68] text-sm px-4 py-3 hover:text-[#171716] transition-colors"
          >
            {step > 1 ? t.back : t.goToLogin}
          </button>

          {step < TOTAL_STEPS ? (
            <button type="button" onClick={handleNext} className="btn flex-1 text-base py-3.5">
              {t.next}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className="btn flex-1 text-base py-3.5 flex items-center justify-center"
            >
              {loading ? <span className="spinner-sm" /> : t.createAccount}
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
      <label className="label block mb-1.5">{label}</label>
      {children}
    </div>
  );
}

function RadioTile({ selected, onClick, title, desc }: {
  selected: boolean; onClick: () => void; title: string; desc: string;
}) {
  return (
    <button type="button" onClick={onClick} className={`tile text-start p-4 ${selected ? 'selected' : ''}`}>
      <div className="flex items-center gap-3 mb-1">
        <span className={`w-2.5 h-2.5 rounded-[2px] flex-shrink-0 ${selected ? 'bg-[#0e7a4a]' : 'border border-[#cfceca]'}`} />
        <span className="text-[#171716] font-bold">{title}</span>
      </div>
      <p className="text-[#6f6e68] text-sm ms-[22px]">{desc}</p>
    </button>
  );
}

function IconTile({ selected, onClick, label }: {
  selected: boolean; onClick: () => void; label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`tile flex items-center justify-center p-4 min-h-[60px] ${selected ? 'selected' : ''}`}
    >
      <span className={`text-sm font-bold text-center ${selected ? 'text-[#171716]' : 'text-[#6f6e68]'}`}>
        {label}
      </span>
    </button>
  );
}
