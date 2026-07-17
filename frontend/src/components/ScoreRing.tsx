// Circular score gauge — number inside a progress ring.
export default function ScoreRing({ score, size = 46 }: { score: number; size?: number }) {
  const stroke = size / 13;
  const pct = Math.min(100, Math.max(0, score));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const color = pct >= 80 ? 'var(--good)' : pct >= 60 ? 'var(--warn)' : 'var(--bad)';

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="flex-shrink-0">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke={color} strokeWidth={stroke} strokeLinecap="round"
        strokeDasharray={`${(pct / 100) * c} ${c}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text
        x="50%" y="52%" dominantBaseline="central" textAnchor="middle"
        fill="var(--ink)" fontSize={size * 0.34} fontWeight={800}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {Math.round(pct)}
      </text>
    </svg>
  );
}
