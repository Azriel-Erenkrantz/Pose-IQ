import { useState } from 'react';
import type { LiveSessionOutput } from '../api/types';
import { useI18n } from '../i18n';

// A single row inside a `.panel .rows` list — not a standalone card.
export default function SessionRow({ session, onSetWeight }: {
  session: LiveSessionOutput;
  onSetWeight: (weightKg: number) => Promise<void>;
}) {
  const { t, locale, exerciseName } = useI18n();
  const dateStr = new Date(session.date).toLocaleDateString(locale, { month: 'short', day: 'numeric' });
  const [editing, setEditing] = useState(false);
  const [draft, setDraft]     = useState(session.weight_kg?.toString() ?? '');
  const [saving, setSaving]   = useState(false);

  const score = session.overall_score;
  const scoreColor = score >= 80 ? 'var(--good)' : score >= 60 ? 'var(--warn)' : 'var(--bad)';

  async function save() {
    const parsed = parseFloat(draft);
    if (isNaN(parsed) || parsed < 0 || parsed > 300) return;
    setSaving(true);
    try {
      await onSetWeight(parsed);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex items-center justify-between px-4 py-3.5 gap-3">
      <div className="min-w-0">
        <p className="text-[#171716] font-bold text-[15px] leading-tight truncate">
          {exerciseName(session.exercise_id, session.exercise_name)}
        </p>
        <p className="text-[#a09f98] text-xs mt-0.5 num">
          {dateStr} · {session.reps.length} {t.repsWord} · {Math.round(session.duration_seconds)}s
        </p>
      </div>

      <div className="flex items-center gap-3 flex-shrink-0">
        {editing ? (
          <span className="flex items-center gap-1.5" dir="ltr">
            <input
              type="number"
              min={0}
              max={300}
              step={0.5}
              value={draft}
              onChange={e => setDraft(e.target.value)}
              className="w-16 bg-white border border-[#e6e5e1] rounded-md px-2 py-1 text-[#171716] text-xs text-right num focus:outline-none focus:border-[#171716]"
              autoFocus
            />
            <span className="text-[#a09f98] text-xs">{t.kg}</span>
            <button
              onClick={save}
              disabled={saving}
              className="text-accent text-xs font-bold disabled:opacity-50"
            >
              {saving ? '…' : t.save}
            </button>
          </span>
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="btn-ghost text-xs font-bold px-2 py-1 num"
          >
            {session.weight_kg != null ? `${session.weight_kg} ${t.kg}` : t.logWeight}
          </button>
        )}
        <span className="font-black text-2xl num w-12 text-end" style={{ color: scoreColor }}>
          {score.toFixed(0)}
        </span>
      </div>
    </div>
  );
}
