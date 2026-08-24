import type {
  AuthToken, DashboardData, ExerciseDef, HealthStatus, LiveSessionOutput,
  ProfileSetupOptions, ProgressMetrics, RepPayload, SavedSession, User, UserRatings,
} from './types';

// Configurable via .env (VITE_API_BASE) — see .env.example. Falls back to the
// local Flask dev server so `npm run dev` works with zero setup.
// iOS Simulator can reach the Mac's localhost directly via the fallback.
// Physical device: set VITE_API_BASE to your Mac's local IP (e.g. http://192.168.x.x:5000).
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:5000';

// ── Low-level helpers ──────────────────────────────────────────────────────────

// Runs a fetch() and turns every failure mode into a clean, readable Error:
//   - fetch() itself throwing (server unreachable, DNS, CORS) → "Could not
//     reach the server" instead of the browser's raw "Failed to fetch".
//   - a non-JSON response body (e.g. a 500's raw HTML error page) → a
//     status-based message instead of a cryptic JSON-parse SyntaxError.
//   - a JSON error body ({"error": "..."}) → that message, as before.
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error('Could not reach the server. Is it running?');
  }

  let data: any = null;
  try {
    data = await res.json();
  } catch {
    if (!res.ok) throw new Error(`Server error (HTTP ${res.status})`);
    throw new Error('Unexpected response from server');
  }

  if (!res.ok) throw new Error(data?.error ?? `HTTP ${res.status}`);
  return data as T;
}

async function get<T>(path: string, token?: string): Promise<T> {
  return request<T>(path, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
}

async function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function put<T>(path: string, body: unknown, token: string): Promise<T> {
  return request<T>(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
}

async function postAuth<T>(path: string, body: unknown, token: string): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
}

// ── Auth ───────────────────────────────────────────────────────────────────────

export const authApi = {
  register: (body: {
    name: string; email: string; password: string;
    fitness_level?: string; limitations?: string[];
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
  getRatings: (id: string, token: string) => get<UserRatings>(`/api/user/${id}/ratings`, token),
  setRating: (id: string, exerciseId: string, rating: number, token: string) =>
    put<{ user_id: string; exercise_id: string; rating: number }>(
      `/api/user/${id}/ratings/${exerciseId}`, { rating }, token),
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
