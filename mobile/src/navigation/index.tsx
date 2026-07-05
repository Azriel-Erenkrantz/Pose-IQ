import React, { useState } from 'react';
import { AuthToken } from '../api/types';
import DashboardScreen from '../screens/DashboardScreen';
import LoginScreen from '../screens/LoginScreen';

// Simple state-based navigator — no library needed at this stage.
// Replace with React Navigation stack when more screens are added.

type Screen = 'login' | 'dashboard';

export default function AppNavigator() {
  const [screen, setScreen] = useState<Screen>('login');
  const [token, setToken]   = useState<AuthToken | null>(null);

  function handleLogin(t: AuthToken) {
    setToken(t);
    setScreen('dashboard');
  }

  function handleLogout() {
    setToken(null);
    setScreen('login');
  }

  if (screen === 'dashboard') {
    return <DashboardScreen onLogout={handleLogout} />;
  }

  return (
    <LoginScreen
      onLogin={handleLogin}
      onGoToRegister={() => {/* register screen coming next */}}
    />
  );
}
