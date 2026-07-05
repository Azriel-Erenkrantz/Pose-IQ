import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { userApi } from '../api/client';
import {
  AuthToken,
  BodyRegion,
  DashboardData,
  ExerciseRecommendation,
  HealthStatus,
  LiveSessionOutput,
  ProgressMetrics,
} from '../api/types';

const REGIONS: BodyRegion[] = ['upper', 'core', 'lower'];
const REGION_LABEL: Record<BodyRegion, string> = {
  upper: 'Upper Body',
  core:  'Core',
  lower: 'Lower Body',
};

interface Props {
  token: AuthToken;
  onLogout: () => void;
}

export default function DashboardScreen({ token, onLogout }: Props) {
  const [data, setData]             = useState<DashboardData | null>(null);
  const [loading, setLoading]       = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError]           = useState('');

  // Local health rating state (mirrors what we'll PUT to the API)
  const [ratings, setRatings] = useState<Record<BodyRegion, number>>({
    upper: 5, core: 5, lower: 5,
  });
  const [savingHealth, setSavingHealth] = useState(false);
  const [healthSaved, setHealthSaved]   = useState(false);

  const load = useCallback(async (isRefresh = false) => {
    isRefresh ? setRefreshing(true) : setLoading(true);
    setError('');
    try {
      const dashboard = await userApi.getDashboard(token.user_id, token.token);
      setData(dashboard);
      // Seed local rating state from server
      const serverRatings = dashboard.health_status.ratings;
      setRatings({
        upper: serverRatings.upper ?? 5,
        core:  serverRatings.core  ?? 5,
        lower: serverRatings.lower ?? 5,
      });
      console.log('recommendations count:', dashboard.recommendations.length, dashboard.recommendations);
    } catch (e: any) {
      setError('Could not load dashboard. Is the server running?');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  async function saveHealth() {
    setSavingHealth(true);
    setHealthSaved(false);
    try {
      await userApi.updateHealth(token.user_id, ratings, token.token);
      setHealthSaved(true);
      // Refresh recommendations which depend on health
      const dashboard = await userApi.getDashboard(token.user_id, token.token);
      setData(dashboard);
      setTimeout(() => setHealthSaved(false), 2000);
    } catch {
      // keep existing data
    } finally {
      setSavingHealth(false);
    }
  }

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#4ade80" />
      </View>
    );
  }

  if (error || !data) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorText}>{error || 'No data.'}</Text>
      </View>
    );
  }

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} tintColor="#4ade80" />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.greeting}>{greeting},</Text>
          <Text style={styles.userName}>{data.user.name}</Text>
        </View>
        <View style={styles.headerRight}>
          <View style={styles.levelBadge}>
            <Text style={styles.levelText}>{data.user.fitness_level}</Text>
          </View>
          <TouchableOpacity onPress={onLogout} style={styles.logoutBtn}>
            <Text style={styles.logoutText}>Logout</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Stats row */}
      <View style={styles.statsRow}>
        <StatCard label="Sessions"  value={String(data.recent_sessions.length)} unit="recent"  color="#4ade80" />
        <StatCard label="Avg Score" value={avgScore(data.recent_sessions)}       unit="/ 100"   color="#60a5fa" />
        <StatCard label="Exercises" value={String(data.progress_summary.length)} unit="tracked" color="#f59e0b" />
      </View>

      {/* Injury risk banner */}
      {data.injury_risk && data.injury_risk.overall_risk >= 0.4 && (
        <View style={styles.riskBanner}>
          <Text style={styles.riskTitle}>Injury risk detected</Text>
          <Text style={styles.riskText}>{data.injury_risk.recommendation}</Text>
        </View>
      )}

      {/* Health status card */}
      <Text style={styles.sectionTitle}>How do you feel today?</Text>
      <View style={styles.card}>
        {REGIONS.map(region => (
          <View key={region} style={styles.ratingRow}>
            <Text style={styles.regionLabel}>{REGION_LABEL[region]}</Text>
            <View style={styles.ratingButtons}>
              {[1, 2, 3, 4, 5].map(n => {
                const selected = ratings[region] === n;
                const dotColor = n <= 2 ? '#ff6b6b' : n === 3 ? '#f59e0b' : '#4ade80';
                return (
                  <TouchableOpacity
                    key={n}
                    style={[styles.ratingBtn, selected && { backgroundColor: dotColor + '33', borderColor: dotColor }]}
                    onPress={() => setRatings(r => ({ ...r, [region]: n }))}
                  >
                    <Text style={[styles.ratingNum, selected && { color: dotColor }]}>{n}</Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>
        ))}
        <TouchableOpacity
          style={[styles.saveBtn, savingHealth && styles.saveBtnDisabled]}
          onPress={saveHealth}
          disabled={savingHealth}
        >
          <Text style={styles.saveBtnText}>
            {savingHealth ? 'Saving…' : healthSaved ? 'Saved!' : 'Save & refresh recommendations'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Recommendations */}
      <Text style={styles.sectionTitle}>Recommended for you</Text>
      {data.recommendations.length > 0
        ? data.recommendations.slice(0, 5).map(r => (
            <RecommendationCard key={r.exercise.exercise_id} rec={r} />
          ))
        : (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyText}>No recommendations yet.</Text>
            <Text style={styles.emptyHint}>Rate your health above and tap Save, or make sure you selected goals during registration.</Text>
          </View>
        )
      }

      {/* Progress by exercise */}
      {data.progress_summary.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Your Progress</Text>
          {data.progress_summary.map(m => (
            <ProgressCard key={m.exercise_id} metrics={m} />
          ))}
        </>
      )}

      {/* Recent sessions */}
      {data.recent_sessions.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Recent Sessions</Text>
          {data.recent_sessions.map(s => (
            <SessionCard key={s.session_id} session={s} />
          ))}
        </>
      )}
    </ScrollView>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatCard({ label, value, unit, color }: {
  label: string; value: string; unit: string; color: string;
}) {
  return (
    <View style={styles.statCard}>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
      <Text style={styles.statUnit}>{unit}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function RecommendationCard({ rec }: { rec: ExerciseRecommendation }) {
  const scenarioColor =
    rec.scenario === 'all_healthy'           ? '#4ade80' :
    rec.scenario === 'target_healthy_adjacent_not' ? '#f59e0b' : '#60a5fa';
  const scenarioLabel =
    rec.scenario === 'all_healthy'           ? 'Healthy' :
    rec.scenario === 'target_healthy_adjacent_not' ? 'Take it easy' : 'Recovery';

  return (
    <View style={styles.card}>
      <View style={styles.cardRow}>
        <Text style={styles.cardTitle}>{rec.exercise.name}</Text>
        <View style={[styles.scenarioBadge, { borderColor: scenarioColor + '66' }]}>
          <Text style={[styles.scenarioText, { color: scenarioColor }]}>{scenarioLabel}</Text>
        </View>
      </View>
      <Text style={styles.cardMeta}>{rec.reason}</Text>
      <View style={styles.cardRow}>
        <Text style={styles.tagList}>
          {rec.exercise.tags.slice(0, 3).map(t => `#${t}`).join('  ')}
        </Text>
        <Text style={[styles.cardScore, { fontSize: 16 }]}>{(rec.score * 100).toFixed(0)}</Text>
      </View>
      <ScoreBar score={rec.score * 100} />
    </View>
  );
}

function ProgressCard({ metrics }: { metrics: ProgressMetrics }) {
  const trendColor = metrics.score_trend === 'improving' ? '#4ade80' : metrics.score_trend === 'declining' ? '#ff6b6b' : '#888';
  const trendIcon  = metrics.score_trend === 'improving' ? '↑' : metrics.score_trend === 'declining' ? '↓' : '→';

  return (
    <View style={styles.card}>
      <View style={styles.cardRow}>
        <Text style={styles.cardTitle}>{metrics.exercise_name}</Text>
        <Text style={[styles.trendBadge, { color: trendColor }]}>{trendIcon} {metrics.score_trend}</Text>
      </View>
      <View style={styles.cardRow}>
        <Text style={styles.cardMeta}>{metrics.total_sessions} sessions · {metrics.total_reps} reps</Text>
        <Text style={styles.cardScore}>{metrics.avg_score_recent.toFixed(1)}</Text>
      </View>
      <ScoreBar score={metrics.avg_score_recent} />
      {metrics.weak_joints.length > 0 && (
        <Text style={styles.weakJoints}>
          Weak: {metrics.weak_joints.map(([j]) => j.replace(/_/g, ' ')).join(', ')}
        </Text>
      )}
    </View>
  );
}

function SessionCard({ session }: { session: LiveSessionOutput }) {
  const dateStr = new Date(session.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return (
    <View style={styles.card}>
      <View style={styles.cardRow}>
        <Text style={styles.cardTitle}>{session.exercise_name}</Text>
        <Text style={styles.cardScore}>{session.overall_score.toFixed(0)}</Text>
      </View>
      <Text style={styles.cardMeta}>
        {dateStr} · {session.reps.length} reps · {Math.round(session.duration_seconds)}s
      </Text>
      <ScoreBar score={session.overall_score} />
    </View>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct   = Math.min(100, Math.max(0, score));
  const color = pct >= 80 ? '#4ade80' : pct >= 60 ? '#f59e0b' : '#ff6b6b';
  return (
    <View style={styles.barBg}>
      <View style={[styles.barFill, { width: `${pct}%` as any, backgroundColor: color }]} />
    </View>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function avgScore(sessions: LiveSessionOutput[]): string {
  if (!sessions.length) return '—';
  return (sessions.reduce((s, r) => s + r.overall_score, 0) / sessions.length).toFixed(1);
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f1a' },
  content:   { padding: 20, paddingBottom: 40 },
  centered:  { flex: 1, backgroundColor: '#0f0f1a', justifyContent: 'center', alignItems: 'center' },
  errorText: { color: '#ff6b6b', textAlign: 'center', paddingHorizontal: 32 },

  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
    marginTop: 12,
  },
  greeting: { color: '#888', fontSize: 14 },
  userName: { color: '#fff', fontSize: 26, fontWeight: '700', marginTop: 2 },
  headerRight: { alignItems: 'flex-end', gap: 8 },
  levelBadge: {
    backgroundColor: '#1a1a2e',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: '#2a2a40',
  },
  levelText:  { color: '#4ade80', fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },
  logoutBtn:  { paddingHorizontal: 10, paddingVertical: 4 },
  logoutText: { color: '#555', fontSize: 12 },

  statsRow: { flexDirection: 'row', gap: 10, marginBottom: 24 },
  statCard: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    borderRadius: 14,
    padding: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a40',
  },
  statValue: { fontSize: 24, fontWeight: '800' },
  statUnit:  { color: '#555', fontSize: 10, marginTop: 1 },
  statLabel: { color: '#888', fontSize: 11, marginTop: 4, fontWeight: '600' },

  riskBanner: {
    backgroundColor: '#2a1a1a',
    borderRadius: 14,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#ff6b6b44',
  },
  riskTitle: { color: '#ff6b6b', fontWeight: '700', fontSize: 14, marginBottom: 6 },
  riskText:  { color: '#ccc', fontSize: 13, lineHeight: 18 },

  sectionTitle: { color: '#fff', fontSize: 18, fontWeight: '700', marginBottom: 12, marginTop: 8 },

  card: {
    backgroundColor: '#1a1a2e',
    borderRadius: 14,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2a2a40',
  },
  cardRow:   { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  cardTitle: { color: '#fff', fontSize: 15, fontWeight: '600', flex: 1 },
  cardMeta:  { color: '#666', fontSize: 12, marginBottom: 10 },
  cardScore: { color: '#4ade80', fontSize: 20, fontWeight: '800' },
  trendBadge:{ fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },
  weakJoints:{ color: '#f59e0b', fontSize: 12, marginTop: 6, textTransform: 'capitalize' },

  scenarioBadge: {
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderWidth: 1,
  },
  scenarioText: { fontSize: 11, fontWeight: '600' },
  tagList: { color: '#555', fontSize: 12 },

  // Health rating
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  regionLabel: { color: '#aaa', fontSize: 13, fontWeight: '600', flex: 1 },
  ratingButtons: { flexDirection: 'row', gap: 6 },
  ratingBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#0f0f1a',
    borderWidth: 1,
    borderColor: '#2a2a40',
    justifyContent: 'center',
    alignItems: 'center',
  },
  ratingNum: { color: '#555', fontSize: 14, fontWeight: '600' },
  saveBtn: {
    marginTop: 8,
    backgroundColor: '#4ade8022',
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#4ade8066',
  },
  saveBtnDisabled: { opacity: 0.5 },
  saveBtnText: { color: '#4ade80', fontSize: 13, fontWeight: '600' },

  emptyCard: {
    backgroundColor: '#1a1a2e',
    borderRadius: 14,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2a2a40',
  },
  emptyText: { color: '#555', fontSize: 14, fontWeight: '600', marginBottom: 4 },
  emptyHint: { color: '#444', fontSize: 12, lineHeight: 18 },

  barBg:   { height: 4, backgroundColor: '#2a2a40', borderRadius: 2, overflow: 'hidden' },
  barFill: { height: 4, borderRadius: 2 },
});
