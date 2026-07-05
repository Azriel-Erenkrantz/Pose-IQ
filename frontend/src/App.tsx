import { useEffect, useState } from 'react';
import type { AuthToken } from './api/types';
import DashboardScreen from './screens/DashboardScreen';
import LoginScreen from './screens/LoginScreen';
import RegisterScreen from './screens/RegisterScreen';

const TOKEN_KEY = 'pose_iq_token';

type Screen = 'loading' | 'login' | 'register' | 'dashboard';

export default function App() {
  const [screen, setScreen] = useState<Screen>('loading');
  const [token, setToken]   = useState<AuthToken | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem(TOKEN_KEY);
    if (raw) {
      const saved: AuthToken = JSON.parse(raw);
      if (new Date(saved.expires_at) > new Date()) {
        setToken(saved);
        setScreen('dashboard');
        return;
      }
    }
    setScreen('login');
  }, []);

  function handleLogin(t: AuthToken) {
    localStorage.setItem(TOKEN_KEY, JSON.stringify(t));
    setToken(t);
    setScreen('dashboard');
  }

  function handleRegister(t: AuthToken) {
    localStorage.setItem(TOKEN_KEY, JSON.stringify(t));
    setToken(t);
    setScreen('dashboard');
  }

  function handleLogout() {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setScreen('login');
  }

  if (screen === 'loading') {
    return (
      <div className="flex flex-1 items-center justify-center bg-[#0f0f1a]">
        <div className="spinner" />
      </div>
    );
  }

  if (screen === 'dashboard' && token) {
    return <DashboardScreen token={token} onLogout={handleLogout} />;
  }

  if (screen === 'register') {
    return <RegisterScreen onRegister={handleRegister} onGoToLogin={() => setScreen('login')} />;
  }

  return <LoginScreen onLogin={handleLogin} onGoToRegister={() => setScreen('register')} />;
}
