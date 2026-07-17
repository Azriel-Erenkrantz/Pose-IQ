import React, { useState } from 'react';
import { authApi } from '../api/client';
import type { AuthToken } from '../api/types';
import { useI18n } from '../i18n';

interface Props {
  onLogin: (token: AuthToken) => void;
  onGoToRegister: () => void;
}

export default function LoginScreen({ onLogin, onGoToRegister }: Props) {
  const { t, lang, setLang } = useI18n();
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    if (!email || !password) { setError(t.fillBothFields); return; }
    setLoading(true);
    setError('');
    try {
      const token = await authApi.login(email.trim().toLowerCase(), password);
      onLogin(token);
    } catch (err: any) {
      setError(err.message ?? t.loginFailed);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen px-6">
      <button
        onClick={() => setLang(lang === 'he' ? 'en' : 'he')}
        className="btn-ghost absolute top-5 end-5 text-xs font-bold px-2.5 py-1.5"
      >
        {lang === 'he' ? 'EN' : 'עב'}
      </button>

      <div className="w-full max-w-sm">
        {/* Wordmark */}
        <div className="mb-12">
          <h1 className="text-5xl font-black tracking-tight" dir="ltr">
            POSE-IQ<span className="text-accent">.</span>
          </h1>
          <p className="label mt-3">{t.tagline}</p>
        </div>

        {/* Form */}
        <form onSubmit={handleLogin} className="flex flex-col gap-4">
          <div>
            <label className="label block mb-1.5">{t.email}</label>
            <input
              type="email"
              className="field"
              placeholder="you@example.com"
              dir="ltr"
              value={email}
              onChange={e => setEmail(e.target.value)}
              autoComplete="email"
            />
          </div>

          <div>
            <label className="label block mb-1.5">{t.password}</label>
            <input
              type="password"
              className="field"
              placeholder="••••••••"
              dir="ltr"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>

          {error && <p className="text-[#d92d20] text-sm">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="btn mt-2 text-base py-3.5 flex items-center justify-center"
          >
            {loading ? <span className="spinner-sm" /> : t.loginBtn}
          </button>

          <button
            type="button"
            onClick={onGoToRegister}
            className="text-[#6f6e68] text-sm text-center hover:text-[#171716] transition-colors mt-1"
          >
            {t.noAccount} <span className="text-accent font-bold">{t.createOne}</span>
          </button>
        </form>
      </div>
    </div>
  );
}
