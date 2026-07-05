import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { devApi } from '../api/client';
import { DashboardData, LiveSessionOutput, ProgressMetrics } from '../api/types';

interface Props {
  onLogout: () => void;
}

export default function DashboardScreen({ onLogout }: Props) {
  const [data, setData]           = useState<DashboardData | null>(null);
  const [loading, setLoading]     = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError]         = useState('');

  const load = useCallback(async (isRefresh = false) => {
    isRefresh ? setRefreshing(true) : setLoading(true);
    setError('');
    try {
      const dashboard = await devApi.dashboard();
      setData(dashboard);
    } catch (e: any) {
      setError('Could not load dashboard. Is the server running?');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

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
        <View style={styles.levelBadge}>
          <Text style={styles.levelText}>{data.user.fitness_level}</Text>
        </View>
      </View>

      {/* Stats row */}
      <View style={styles.statsRow}>
        <StatCard
          label="Sessions"
          value={String(data.recent_sessions.length)}
          unit="recent"
          color="#4ade80"
        />
        <StatCard
          label="Avg Score"
          value={avgScore(data.recent_sessions)}
          unit="/ 100"
          color="#60a5fa"
        />
        <StatCard
          label="Exercises"
          value={String(data.progress_summary.length)}
          unit="tracked"
          color="#f59e0b"
        />
      </View>

      {/* Injury risk banner */}
      {data.injury_risk && data.injury_risk.overall_risk >= 0.4 && (
        <View style={styles.riskBanner}>
          <Text style={styles.riskTitle}>⚠ Injury risk detected</Text>
          <Text style={styles.riskText}>{data.injury_risk.recommendation}</Text>
        </View>
      )}

      {/* Progress by exercise */}
      <Text style={styles.sectionTitle}>Your Progress</Text>
      {data.progress_summary.map(m => (
        <ProgressCard key={m.exercise_id} metrics={m} />
      ))}

      {/* Recent sessions */}
      <Text style={styles.sectionTitle}>Recent Sessions</Text>
      {data.recent_sessions.map(s => (
        <SessionCard key={s.session_id} session={s} />
      ))}
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

function ProgressCard({ metrics }: { metrics: ProgressMetrics }) {
  const trendColor = metrics.score_trend === 'improving'
    ? '#4ade80'
    : metrics.score_trend === 'declining'
    ? '#ff6b6b'
    : '#888';
  const trendIcon = metrics.score_trend === 'improving' ? '↑' : metrics.score_trend === 'declining' ? '↓' : '→';

  return (
    <View style={styles.card}>
      <View style={styles.cardRow}>
        <Text style={styles.cardTitle}>{metrics.exercise_name}</Text>
        <Text style={[styles.trendBadge, { color: trendColor }]}>
          {trendIcon} {metrics.score_trend}
        </Text>
      </View>
      <View style={styles.cardRow}>
        <Text style={styles.cardMeta}>
          {metrics.total_sessions} sessions · {metrics.total_reps} reps
        </Text>
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
  const date = new Date(session.date);
  const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

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
  const pct = Math.min(100, Math.max(0, score));
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
  const avg = sessions.reduce((s, r) => s + r.overall_score, 0) / sessions.length;
  return avg.toFixed(1);
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
  levelBadge: {
    backgroundColor: '#1a1a2e',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: '#2a2a40',
  },
  levelText: { color: '#4ade80', fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },

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
  cardTitle: { color: '#fff', fontSize: 15, fontWeight: '600' },
  cardMeta:  { color: '#666', fontSize: 12, marginBottom: 10 },
  cardScore: { color: '#4ade80', fontSize: 20, fontWeight: '800' },
  trendBadge:{ fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },
  weakJoints:{ color: '#f59e0b', fontSize: 12, marginTop: 6, textTransform: 'capitalize' },

  barBg:   { height: 4, backgroundColor: '#2a2a40', borderRadius: 2, overflow: 'hidden' },
  barFill: { height: 4, borderRadius: 2 },
});
