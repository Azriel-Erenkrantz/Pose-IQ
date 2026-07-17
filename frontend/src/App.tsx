import { useEffect, useState } from 'react';
import type { AuthToken } from './api/types';
import NavBar from './components/NavBar';
import type { Tab } from './components/NavBar';
import { LanguageProvider } from './i18n';
import DashboardScreen from './screens/DashboardScreen';
import HistoryScreen from './screens/HistoryScreen';
import LoginScreen from './screens/LoginScreen';
import RegisterScreen from './screens/RegisterScreen';
import WorkoutScreen from './screens/WorkoutScreen';

const TOKEN_KEY = 'pose_iq_token';

type Screen = 'loading' | 'login' | 'register' | 'app';

export default function App() {
  return (
    <LanguageProvider>
      <Shell />
    </LanguageProvider>
  );
}

function Shell() {
  const [screen, setScreen] = useState<Screen>('loading');
  const [tab, setTab]       = useState<Tab>('home');
  const [token, setToken]   = useState<AuthToken | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem(TOKEN_KEY);
    if (raw) {
      const saved: AuthToken = JSON.parse(raw);
      if (new Date(saved.expires_at) > new Date()) {
        setToken(saved);
        setScreen('app');
        return;
      }
    }
    setScreen('login');
  }, []);

  function handleAuth(t: AuthToken) {
    localStorage.setItem(TOKEN_KEY, JSON.stringify(t));
    setToken(t);
    setTab('home');
    setScreen('app');
  }

  function handleLogout() {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setScreen('login');
  }

  if (screen === 'loading') {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="spinner" />
      </div>
    );
  }

  if (screen === 'app' && token) {
    return (
      <div className="flex flex-col min-h-screen">
        <NavBar active={tab} onNavigate={setTab} onLogout={handleLogout} />
        {tab === 'home'    && <DashboardScreen token={token} onNavigate={setTab} />}
        {tab === 'workout' && <WorkoutScreen token={token} onNavigate={setTab} />}
        {tab === 'history' && <HistoryScreen token={token} />}
      </div>
    );
  }

  if (screen === 'register') {
    return <RegisterScreen onRegister={handleAuth} onGoToLogin={() => setScreen('login')} />;
  }

  return <LoginScreen onLogin={handleAuth} onGoToRegister={() => setScreen('register')} />;
}
