import type {
  AuthToken, DashboardData, ExerciseDef, HealthStatus, LiveSessionOutput,
  ProfileSetupOptions, ProgressMetrics, RepPayload, SavedSession, User,
} from './types';

// iOS Simulator can reach the Mac's localhost directly.
// Physical device: replace with your Mac's local IP (e.g. http://192.168.x.x:5001)
const API_BASE = 'http://localhost:5001';

// ── Low-level helpers ──────────────────────────────────────────────────────────

async function get<T>(path: string, token?: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data as T;
}

async function put<T>(path: string, body: unknown, token: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data as T;
}

async function postAuth<T>(path: string, body: unknown, token: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data as T;
}

// ── Auth ───────────────────────────────────────────────────────────────────────

export const authApi = {
  register: (body: {
    name: string; email: string; password: string;
    fitness_level?: string; trainer_personality?: string;
    target_goals?: string[]; equipment?: string[]; limitations?: string[];
  }) => post<AuthToken>('/api/auth/register', body),

  login: (email: string, password: string) =>
    post<AuthToken>('/api/auth/login', { email, password }),

  options: () => get<ProfileSetupOptions>('/api/auth/options'),
};

// ── User ───────────────────────────────────────────────────────────────────────

export const userApi = {
  getUser:    (id: string, token: string) => get<User>(`/api/user/${id}`, token),
  updateUser: (id: string, body: Partial<User>, token: string) =>
    put<User>(`/api/user/${id}`, body, token),
  getDashboard: (id: string, token: string) =>
    get<DashboardData>(`/api/dashboard/${id}`, token),
  getHistory: (id: string, token: string) =>
    get<LiveSessionOutput[]>(`/api/user/${id}/history`, token),
  getProgress: (id: string, token: string) =>
    get<ProgressMetrics[]>(`/api/user/${id}/progress`, token),
  updateHealth: (id: string, ratings: Record<string, number>, token: string) =>
    put<HealthStatus>(`/api/user/${id}/health`, ratings, token),
  setSessionWeight: (id: string, sessionId: string, weightKg: number | null, token: string) =>
    put<{ session_id: string; weight_kg: number | null }>(
      `/api/user/${id}/sessions/${sessionId}/weight`, { weight_kg: weightKg }, token),
  getExercises: () => get<ExerciseDef[]>('/api/exercises'),
  saveWorkoutSession: (
    id: string,
    body: {
      exercise_id: string;
      exercise_name?: string;
      duration_seconds: number;
      weight_kg?: number | null;
      reps: RepPayload[];
    },
    token: string,
  ) => postAuth<SavedSession>(`/api/user/${id}/sessions`, body, token),
};

// ── Dev / fake-data (no auth needed, for UI development) ──────────────────────

export const devApi = {
  dashboard: () => get<DashboardData>('/api/dev/dashboard'),
  options:   () => get<ProfileSetupOptions>('/api/dev/options'),
};
