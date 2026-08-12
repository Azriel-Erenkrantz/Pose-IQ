import { useCallback, useEffect, useState } from 'react';
import { userApi } from '../api/client';
import type {
  AuthToken,
  BodyRegion,
  DashboardData,
  LiveSessionOutput,
} from '../api/types';
import ScoreRing from '../components/ScoreRing';
import SessionRow from '../components/SessionCard';
import type { Tab } from '../components/NavBar';
import { useI18n } from '../i18n';

const REGIONS: BodyRegion[] = ['upper', 'core', 'lower'];

interface Props {
  token: AuthToken;
  onNavigate: (tab: Tab) => void;
}

export default function DashboardScreen({ token, onNavigate }: Props) {
  const { t, exerciseName, formatReason } = useI18n();
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
    } catch {
      setError(t.couldNotLoad);
    } finally {
      setLoading(false);
    }
  }, [token, t]);

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
      <div className="flex flex-1 items-center justify-center">
        <div className="spinner" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-1 items-center justify-center px-8">
        <p className="text-[#d92d20] text-center">{error || t.couldNotLoad}</p>
      </div>
    );
  }

  const hour = new Date().getHours();
  const greeting = hour < 12 ? t.greetingMorning : hour < 17 ? t.greetingAfternoon : t.greetingEvening;

  return (
    <div className="max-w-2xl w-full mx-auto px-5 py-7 pb-16">

      {/* Header */}
      <div className="flex items-end justify-between mb-7">
        <div>
          <p className="label">{greeting}</p>
          <h1 className="text-4xl font-black text-[#171716] leading-tight">{data.user.name}</h1>
        </div>
        <span className="label pb-1.5">{t.fitnessLabels[data.user.fitness_level]}</span>
      </div>

      {/* Stat strip */}
      <div className="panel flex mb-8">
        <Stat value={String(data.recent_sessions.length)} caption={`${t.statSessions} · ${t.statRecent}`} />
        <div className="w-px bg-[#e6e5e1] my-3" />
        <div className="flex-1 px-4 py-4 flex items-center gap-3">
          {avgScoreNum(data.recent_sessions) !== null
            ? <ScoreRing score={avgScoreNum(data.recent_sessions)!} size={44} />
            : <p className="text-[#171716] font-black text-3xl num leading-none">—</p>}
          <p className="label">{t.statAvgScore}<br />{t.statOutOf}</p>
        </div>
        <div className="w-px bg-[#e6e5e1] my-3" />
        <Stat value={String(data.progress_summary.length)} caption={`${t.statExercises} · ${t.statTracked}`} />
      </div>

      {/* Injury risk */}
      {data.injury_risk && data.injury_risk.overall_risk >= 0.4 && (
        <div className="panel border-[#d92d2055] p-4 mb-8">
          <p className="text-[#d92d20] font-bold text-sm mb-1">{t.injuryRisk}</p>
          <p className="text-[#6f6e68] text-sm">{data.injury_risk.recommendation}</p>
        </div>
      )}

      {/* Health status */}
      <p className="label mb-2.5">{t.howDoYouFeel}</p>
      <div className="panel p-4 mb-8">
        {REGIONS.map(region => (
          <div key={region} className="flex items-center justify-between gap-4 mb-3.5 last:mb-0">
            <span className="text-[#6f6e68] text-sm font-bold w-32 flex-shrink-0">{t.regionLabels[region]}</span>
            <div className="flex flex-1 max-w-[240px] border border-[#e6e5e1] rounded-lg overflow-hidden" dir="ltr">
              {[1, 2, 3, 4, 5].map(n => {
                const selected = ratings[region] === n;
                return (
                  <button
                    key={n}
                    onClick={() => setRatings(r => ({ ...r, [region]: n }))}
                    className={`flex-1 py-1.5 text-sm font-bold num transition-colors border-l first:border-l-0 border-[#e6e5e1] ${
                      selected ? 'bg-[#171716] text-white' : 'text-[#a09f98] hover:text-[#171716]'
                    }`}
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
          className="btn mt-4 w-full text-sm py-2.5 flex items-center justify-center gap-2"
        >
          {savingHealth
            ? <><span className="spinner-sm" /> {t.saving}</>
            : healthSaved ? t.savedBang : t.saveHealth}
        </button>
      </div>

      {/* Recommendations */}
      <p className="label mb-2.5">{t.recommendedForYou}</p>
      {data.recommendations.length > 0 ? (
        <div className="panel rows mb-8">
          {data.recommendations.slice(0, 5).map(r => {
            const scenarioLabel =
              r.scenario === 'all_healthy' ? t.scenarioHealthy :
              r.scenario === 'target_healthy_adjacent_not' ? t.scenarioTakeItEasy : t.scenarioRecovery;
            return (
              <div key={r.exercise.exercise_id} className="flex items-center justify-between px-4 py-3.5 gap-3">
                <div className="min-w-0">
                  <p className="text-[#171716] font-bold text-[15px] leading-tight">
                    {exerciseName(r.exercise.exercise_id, r.exercise.name)}
                  </p>
                  <p className="text-[#a09f98] text-xs mt-0.5 truncate">
                    {formatReason(r.reason_code, r.reason_params, r.reason)}
                  </p>
                </div>
                <div className="text-end flex-shrink-0">
                  <p className="text-accent font-black text-2xl num leading-none">{(r.score * 100).toFixed(0)}</p>
                  <p className="label mt-1">{scenarioLabel}</p>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="panel p-4 mb-8">
          <p className="text-[#6f6e68] text-sm font-bold mb-1">{t.noRecsTitle}</p>
          <p className="text-[#a09f98] text-xs">{t.noRecsHint}</p>
        </div>
      )}

      {/* Weight recommendations */}
      {data.weight_recommendations.length > 0 && (
        <>
          <p className="label mb-2.5">{t.nextWeight}</p>
          <div className="panel rows mb-8">
            {data.weight_recommendations.map(w => {
              const weightLabel = w.recommended_weight_kg > 0
                ? `${w.recommended_weight_kg % 1 === 0 ? w.recommended_weight_kg.toFixed(0) : w.recommended_weight_kg.toFixed(1)}`
                : null;
              return (
                <div key={w.exercise_id} className="flex items-center justify-between px-4 py-3.5 gap-3">
                  <div className="min-w-0">
                    <p className="text-[#171716] font-bold text-[15px] leading-tight">
                      {exerciseName(w.exercise_id, w.exercise_name)}
                    </p>
                    <p className="text-[#a09f98] text-xs mt-0.5">
                      {formatReason(w.reasoning_code, w.reasoning_params, w.reasoning)}
                    </p>
                  </div>
                  <div className="text-end flex-shrink-0">
                    {weightLabel ? (
                      <p className="text-[#171716] font-black text-2xl num leading-none">
                        {weightLabel} <span className="label">{t.kg}</span>
                      </p>
                    ) : (
                      <p className="text-[#171716] font-black text-sm leading-none">{t.bodyweight}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Progress */}
      {data.progress_summary.length > 0 && (
        <>
          <p className="label mb-2.5">{t.yourProgress}</p>
          <div className="panel rows mb-8">
            {data.progress_summary.map(m => (
              <div key={m.exercise_id} className="flex items-center justify-between px-4 py-3.5 gap-3">
                <div className="min-w-0">
                  <p className="text-[#171716] font-bold text-[15px] leading-tight">
                    {exerciseName(m.exercise_id, m.exercise_name)}
                  </p>
                  <p className="text-[#a09f98] text-xs mt-0.5 num">
                    {m.total_sessions} {t.sessionsWord} · {m.total_reps} {t.repsWord} · {t.trendLabels[m.score_trend]}
                  </p>
                  {m.weak_joints.length > 0 && (
                    <p className="text-[#b45309] text-xs mt-0.5 capitalize">
                      {t.weakPrefix}: {m.weak_joints.map(([j]) => j.replace(/_/g, ' ')).join(', ')}
                    </p>
                  )}
                </div>
                <ScoreRing score={m.avg_score_recent} />
              </div>
            ))}
          </div>
        </>
      )}

      {/* Recent sessions */}
      {data.recent_sessions.length > 0 && (
        <>
          <div className="flex items-center justify-between mb-2.5">
            <p className="label">{t.recentSessions}</p>
            <button
              onClick={() => onNavigate('history')}
              className="text-accent text-xs font-bold hover:text-[#171716] transition-colors"
            >
              {t.viewAll}
            </button>
          </div>
          <div className="panel rows">
            {data.recent_sessions.slice(0, 3).map(s => (
              <SessionRow
                key={s.session_id}
                session={s}
                onSetWeight={async (weightKg) => {
                  await userApi.setSessionWeight(token.user_id, s.session_id, weightKg, token.token);
                  await load();
                }}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ value, caption }: { value: string; caption: string }) {
  return (
    <div className="flex-1 px-4 py-4">
      <p className="text-[#171716] font-black text-3xl num leading-none">{value}</p>
      <p className="label mt-1.5">{caption}</p>
    </div>
  );
}

function avgScoreNum(sessions: LiveSessionOutput[]): number | null {
  if (!sessions.length) return null;
  return sessions.reduce((s, r) => s + r.overall_score, 0) / sessions.length;
}
