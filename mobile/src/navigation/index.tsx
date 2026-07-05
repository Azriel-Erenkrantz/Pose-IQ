import React, { useState } from 'react';
import { AuthToken } from '../api/types';
import DashboardScreen from '../screens/DashboardScreen';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';

type Screen = 'login' | 'register' | 'dashboard';

export default function AppNavigator() {
  const [screen, setScreen] = useState<Screen>('login');
  const [token, setToken]   = useState<AuthToken | null>(null);

  function handleLogin(t: AuthToken) {
    setToken(t);
    setScreen('dashboard');
  }

  function handleRegister(t: AuthToken) {
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
  if (screen === 'register') {
    return (
      <RegisterScreen
        onRegister={handleRegister}
        onGoToLogin={() => setScreen('login')}
      />
    );
  }
  return (
    <LoginScreen
      onLogin={handleLogin}
      onGoToRegister={() => setScreen('register')}
    />
  );
}
