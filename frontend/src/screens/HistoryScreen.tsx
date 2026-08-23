import { useCallback, useEffect, useState } from 'react';
import { userApi } from '../api/client';
import type { AuthToken, LiveSessionOutput, ProgressMetrics, UserRatings } from '../api/types';
import ScoreRing from '../components/ScoreRing';
import SessionRow from '../components/SessionCard';
import { useI18n } from '../i18n';

interface Props {
  token: AuthToken;
}

export default function HistoryScreen({ token }: Props) {
  const { t, exerciseName } = useI18n();
  const [sessions, setSessions] = useState<LiveSessionOutput[] | null>(null);
  const [progress, setProgress] = useState<ProgressMetrics[]>([]);
  const [ratings, setRatings]   = useState<UserRatings>({});
  const [error, setError]       = useState('');

  const load = useCallback(async () => {
    setError('');
    try {
      const [hist, prog, rat] = await Promise.all([
        userApi.getHistory(token.user_id, token.token),
        userApi.getProgress(token.user_id, token.token),
        userApi.getRatings(token.user_id, token.token),
      ]);
      setSessions(hist);
      setProgress(prog);
      setRatings(rat);
    } catch {
      setError(t.couldNotLoad);
    }
  }, [token, t]);

  useEffect(() => { load(); }, [load]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center px-8">
        <p className="text-[#d92d20] text-center">{error}</p>
      </div>
    );
  }

  if (sessions === null) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl w-full mx-auto px-5 py-7 pb-16">
      <h1 className="text-4xl font-black text-[#171716] mb-7">{t.historyTitle}</h1>

      {sessions.length === 0 ? (
        <div className="panel p-6">
          <p className="text-[#6f6e68] font-bold mb-1">{t.noSessionsYet}</p>
          <p className="text-[#a09f98] text-sm">{t.noSessionsHint}</p>
        </div>
      ) : (
        <>
          {/* Per-exercise progress */}
          {progress.length > 0 && (
            <>
              <p className="label mb-2.5">{t.yourProgress}</p>
              <div className="panel rows mb-8">
                {progress.map(m => (
                  <div key={m.exercise_id} className="flex items-center justify-between px-4 py-3.5 gap-3">
                    <div className="min-w-0">
                      <p className="text-[#171716] font-bold text-[15px] leading-tight">
                        {exerciseName(m.exercise_id, m.exercise_name)}
                      </p>
                      <p className="text-[#a09f98] text-xs mt-0.5 num">
                        {m.total_sessions} {t.sessionsWord} · {m.total_reps} {t.repsWord} · {t.trendLabels[m.score_trend]}
                      </p>
                    </div>
                    <ScoreRing score={m.avg_score_recent} />
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Full session list */}
          <p className="label mb-2.5">{t.allSessions}</p>
          <div className="panel rows">
            {sessions.map(s => (
              <SessionRow
                key={s.session_id}
                session={s}
                myRating={ratings[s.exercise_id]}
                onSetWeight={async (weightKg) => {
                  await userApi.setSessionWeight(token.user_id, s.session_id, weightKg, token.token);
                  await load();
                }}
                onSetRating={async (rating) => {
                  await userApi.setRating(token.user_id, s.exercise_id, rating, token.token);
                }}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
