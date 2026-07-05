import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import { AuthToken } from '../api/types';
import DashboardScreen from '../screens/DashboardScreen';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';

const TOKEN_KEY = 'auth_token';

type Screen = 'loading' | 'login' | 'register' | 'dashboard';

export default function AppNavigator() {
  const [screen, setScreen] = useState<Screen>('loading');
  const [token, setToken]   = useState<AuthToken | null>(null);

  useEffect(() => {
    AsyncStorage.getItem(TOKEN_KEY)
      .then(raw => {
        if (raw) {
          const saved: AuthToken = JSON.parse(raw);
          if (new Date(saved.expires_at) > new Date()) {
            setToken(saved);
            setScreen('dashboard');
            return;
          }
        }
        setScreen('login');
      })
      .catch(() => setScreen('login'));
  }, []);

  async function handleLogin(t: AuthToken) {
    await AsyncStorage.setItem(TOKEN_KEY, JSON.stringify(t));
    setToken(t);
    setScreen('dashboard');
  }

  async function handleRegister(t: AuthToken) {
    await AsyncStorage.setItem(TOKEN_KEY, JSON.stringify(t));
    setToken(t);
    setScreen('dashboard');
  }

  async function handleLogout() {
    await AsyncStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setScreen('login');
  }

  if (screen === 'loading') {
    return (
      <View style={{ flex: 1, backgroundColor: '#0f0f1a', justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#4ade80" />
      </View>
    );
  }

  if (screen === 'dashboard' && token) {
    return <DashboardScreen token={token} onLogout={handleLogout} />;
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
