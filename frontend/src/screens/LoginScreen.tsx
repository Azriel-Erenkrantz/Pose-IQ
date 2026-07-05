import React, { useState } from 'react';
import { authApi } from '../api/client';
import type { AuthToken } from '../api/types';

interface Props {
  onLogin: (token: AuthToken) => void;
  onGoToRegister: () => void;
}

export default function LoginScreen({ onLogin, onGoToRegister }: Props) {
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    if (!email || !password) { setError('Please fill in both fields.'); return; }
    setLoading(true);
    setError('');
    try {
      const token = await authApi.login(email.trim().toLowerCase(), password);
      onLogin(token);
    } catch (err: any) {
      setError(err.message ?? 'Login failed. Check your credentials.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6 bg-[#0f0f1a]">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-extrabold tracking-wide text-white">Pose-IQ</h1>
          <p className="text-[#888] mt-2 text-sm">Your AI fitness coach</p>
        </div>

        {/* Form */}
        <form onSubmit={handleLogin} className="flex flex-col gap-4">
          <div>
            <label className="block text-[#aaa] text-xs font-semibold mb-1 tracking-wide uppercase">Email</label>
            <input
              type="email"
              className="w-full bg-[#1a1a2e] text-white border border-[#2a2a40] rounded-xl px-4 py-3 text-base focus:outline-none focus:border-[#4ade80]"
              placeholder="you@example.com"
              value={email}
              onChange={e => setEmail(e.target.value)}
              autoComplete="email"
            />
          </div>

          <div>
            <label className="block text-[#aaa] text-xs font-semibold mb-1 tracking-wide uppercase">Password</label>
            <input
              type="password"
              className="w-full bg-[#1a1a2e] text-white border border-[#2a2a40] rounded-xl px-4 py-3 text-base focus:outline-none focus:border-[#4ade80]"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>

          {error && <p className="text-[#ff6b6b] text-sm text-center">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="mt-2 bg-[#4ade80] text-[#0f0f1a] font-bold text-base rounded-xl py-4 hover:bg-[#22c55e] transition-colors disabled:opacity-60 flex items-center justify-center"
          >
            {loading ? <span className="spinner-sm" /> : 'Log In'}
          </button>

          <button
            type="button"
            onClick={onGoToRegister}
            className="text-[#888] text-sm text-center hover:text-[#4ade80] transition-colors"
          >
            No account? <span className="text-[#4ade80] font-semibold">Create one</span>
          </button>
        </form>
      </div>
    </div>
  );
}
