import { useCallback, useEffect, useState } from 'react';
import { userApi } from '../api/client';
import type {
  AuthToken,
  BodyRegion,
  DashboardData,
  ExerciseRecommendation,
  LiveSessionOutput,
  ProgressMetrics,
} from '../api/types';

const REGIONS: BodyRegion[] = ['upper', 'core', 'lower'];
const REGION_LABEL: Record<BodyRegion, string> = { upper: 'Upper Body', core: 'Core', lower: 'Lower Body' };

interface Props {
  token: AuthToken;
  onLogout: () => void;
}

export default function DashboardScreen({ token, onLogout }: Props) {
  const [data, setData]         = useState<DashboardData | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');
  const [ratings, setRatings]   = useState<Record<BodyRegion, number>>({ upper: 5, core: 5, lower: 5 });
  const [savingHealth, setSavingHealth] = useState(false);
  const [healthSaved, setHealthSaved]   = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const dashboard = await userApi.getDashboard(token.user_id, token.token);
      setData(dashboard);
      const r = dashboard.health_status.ratings;
      setRatings({ upper: r.upper ?? 5, core: r.core ?? 5, lower: r.lower ?? 5 });
    } catch (e: any) {
      setError('Could not load dashboard. Is the server running?');
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  async function saveHealth() {
    setSavingHealth(true);
    setHealthSaved(false);
    try {
      await userApi.updateHealth(token.user_id, ratings, token.token);
      const dashboard = await userApi.getDashboard(token.user_id, token.token);
      setData(dashboard);
      setHealthSaved(true);
      setTimeout(() => setHealthSaved(false), 2000);
    } finally {
      setSavingHealth(false);
    }
  }

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center bg-[#0f0f1a]">
        <div className="spinner" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-1 items-center justify-center bg-[#0f0f1a] px-8">
        <p className="text-[#ff6b6b] text-center">{error || 'No data.'}</p>
      </div>
    );
  }

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  return (
    <div className="min-h-screen bg-[#0f0f1a]">
      <div className="max-w-2xl mx-auto px-5 py-6 pb-16">

        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <p className="text-[#888] text-sm">{greeting},</p>
            <h1 className="text-3xl font-extrabold text-white mt-1">{data.user.name}</h1>
          </div>
          <div className="flex flex-col items-end gap-2">
            <span className="bg-[#1a1a2e] border border-[#2a2a40] rounded-full px-4 py-1 text-[#4ade80] text-xs font-semibold capitalize">
              {data.user.fitness_level}
            </span>
            <button onClick={onLogout} className="text-[#555] text-xs hover:text-white transition-colors">
              Logout
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          <StatCard label="Sessions"  value={String(data.recent_sessions.length)} unit="recent"  color="#4ade80" />
          <StatCard label="Avg Score" value={avgScore(data.recent_sessions)}       unit="/ 100"   color="#60a5fa" />
          <StatCard label="Exercises" value={String(data.progress_summary.length)} unit="tracked" color="#f59e0b" />
        </div>

        {/* Injury risk */}
        {data.injury_risk && data.injury_risk.overall_risk >= 0.4 && (
          <div className="bg-[#2a1a1a] border border-[#ff6b6b44] rounded-xl p-4 mb-6">
            <p className="text-[#ff6b6b] font-bold text-sm mb-1">Injury risk detected</p>
            <p className="text-[#ccc] text-sm">{data.injury_risk.recommendation}</p>
          </div>
        )}

        {/* Health status */}
        <h2 className="text-white font-bold text-lg mb-3">How do you feel today?</h2>
        <div className="bg-[#1a1a2e] border border-[#2a2a40] rounded-xl p-4 mb-6">
          {REGIONS.map(region => (
            <div key={region} className="flex items-center justify-between mb-4 last:mb-0">
              <span className="text-[#aaa] text-sm font-semibold w-28">{REGION_LABEL[region]}</span>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map(n => {
                  const selected = ratings[region] === n;
                  const color = n <= 2 ? '#ff6b6b' : n === 3 ? '#f59e0b' : '#4ade80';
                  return (
                    <button
                      key={n}
                      onClick={() => setRatings(r => ({ ...r, [region]: n }))}
                      className="w-9 h-9 rounded-full border text-sm font-semibold transition-all"
                      style={{
                        borderColor: selected ? color : '#2a2a40',
                        backgroundColor: selected ? color + '22' : '#0f0f1a',
                        color: selected ? color : '#555',
                      }}
                    >
                      {n}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
          <button
            onClick={saveHealth}
            disabled={savingHealth}
            className="mt-4 w-full bg-[#4ade8022] border border-[#4ade8066] text-[#4ade80] text-sm font-semibold rounded-xl py-2.5 hover:bg-[#4ade8033] transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {savingHealth
              ? <><span className="spinner-sm" /> Saving…</>
              : healthSaved ? 'Saved!' : 'Save & refresh recommendations'}
          </button>
        </div>

        {/* Recommendations */}
        {data.recommendations.length > 0 && (
          <>
            <h2 className="text-white font-bold text-lg mb-3">Recommended for you</h2>
            {data.recommendations.slice(0, 5).map(r => (
              <RecommendationCard key={r.exercise.exercise_id} rec={r} />
            ))}
          </>
        )}

        {/* Progress */}
        {data.progress_summary.length > 0 && (
          <>
            <h2 className="text-white font-bold text-lg mb-3 mt-2">Your Progress</h2>
            {data.progress_summary.map(m => <ProgressCard key={m.exercise_id} metrics={m} />)}
          </>
        )}

        {/* Recent sessions */}
        {data.recent_sessions.length > 0 && (
          <>
            <h2 className="text-white font-bold text-lg mb-3 mt-2">Recent Sessions</h2>
            {data.recent_sessions.map(s => <SessionCard key={s.session_id} session={s} />)}
          </>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatCard({ label, value, unit, color }: { label: string; value: string; unit: string; color: string }) {
  return (
    <div className="bg-[#1a1a2e] border border-[#2a2a40] rounded-xl p-4 flex flex-col items-center">
      <span className="text-2xl font-extrabold" style={{ color }}>{value}</span>
      <span className="text-[#555] text-xs mt-0.5">{unit}</span>
      <span className="text-[#888] text-xs mt-1 font-semibold">{label}</span>
    </div>
  );
}

function RecommendationCard({ rec }: { rec: ExerciseRecommendation }) {
  const scenarioColor =
    rec.scenario === 'all_healthy' ? '#4ade80' :
    rec.scenario === 'target_healthy_adjacent_not' ? '#f59e0b' : '#60a5fa';
  const scenarioLabel =
    rec.scenario === 'all_healthy' ? 'Healthy' :
    rec.scenario === 'target_healthy_adjacent_not' ? 'Take it easy' : 'Recovery';

  return (
    <div className="bg-[#1a1a2e] border border-[#2a2a40] rounded-xl p-4 mb-2.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-white font-semibold text-base">{rec.exercise.name}</span>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full border" style={{ color: scenarioColor, borderColor: scenarioColor + '55' }}>
          {scenarioLabel}
        </span>
      </div>
      <p className="text-[#666] text-xs mb-2">{rec.reason}</p>
      <div className="flex items-center justify-between">
        <span className="text-[#555] text-xs">{rec.exercise.tags.slice(0, 3).map(t => `#${t}`).join('  ')}</span>
        <span className="text-[#4ade80] font-extrabold text-base">{(rec.score * 100).toFixed(0)}</span>
      </div>
      <ScoreBar score={rec.score * 100} />
    </div>
  );
}

function ProgressCard({ metrics }: { metrics: ProgressMetrics }) {
  const trendColor = metrics.score_trend === 'improving' ? '#4ade80' : metrics.score_trend === 'declining' ? '#ff6b6b' : '#888';
  const trendIcon  = metrics.score_trend === 'improving' ? '↑' : metrics.score_trend === 'declining' ? '↓' : '→';
  return (
    <div className="bg-[#1a1a2e] border border-[#2a2a40] rounded-xl p-4 mb-2.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-white font-semibold">{metrics.exercise_name}</span>
        <span className="text-xs font-semibold capitalize" style={{ color: trendColor }}>{trendIcon} {metrics.score_trend}</span>
      </div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[#666] text-xs">{metrics.total_sessions} sessions · {metrics.total_reps} reps</span>
        <span className="text-[#4ade80] font-extrabold text-xl">{metrics.avg_score_recent.toFixed(1)}</span>
      </div>
      <ScoreBar score={metrics.avg_score_recent} />
      {metrics.weak_joints.length > 0 && (
        <p className="text-[#f59e0b] text-xs mt-2 capitalize">
          Weak: {metrics.weak_joints.map(([j]) => j.replace(/_/g, ' ')).join(', ')}
        </p>
      )}
    </div>
  );
}

function SessionCard({ session }: { session: LiveSessionOutput }) {
  const dateStr = new Date(session.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return (
    <div className="bg-[#1a1a2e] border border-[#2a2a40] rounded-xl p-4 mb-2.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-white font-semibold">{session.exercise_name}</span>
        <span className="text-[#4ade80] font-extrabold text-xl">{session.overall_score.toFixed(0)}</span>
      </div>
      <p className="text-[#666] text-xs mb-2">
        {dateStr} · {session.reps.length} reps · {Math.round(session.duration_seconds)}s
      </p>
      <ScoreBar score={session.overall_score} />
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct   = Math.min(100, Math.max(0, score));
  const color = pct >= 80 ? '#4ade80' : pct >= 60 ? '#f59e0b' : '#ff6b6b';
  return (
    <div className="h-1 bg-[#2a2a40] rounded-full overflow-hidden">
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

function avgScore(sessions: LiveSessionOutput[]): string {
  if (!sessions.length) return '—';
  return (sessions.reduce((s, r) => s + r.overall_score, 0) / sessions.length).toFixed(1);
}
