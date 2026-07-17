// Stylized side-view squat skeleton — the same joints/angles the pipeline
// tracks, as a brand illustration: ink limbs, green joints, angle chips.
export default function PoseFigure() {
  const joints: [number, number][] = [
    [116, 62],   // shoulder
    [150, 72],   // elbow
    [182, 66],   // wrist
    [104, 118],  // hip
    [140, 150],  // knee
    [118, 192],  // ankle
  ];

  return (
    <svg viewBox="0 0 240 220" className="w-full max-w-[340px] mx-auto block" style={{ direction: 'ltr' }}>
      {/* tracking mesh */}
      <g stroke="var(--accent)" strokeWidth="1" opacity=".3">
        <line x1="182" y1="66" x2="212" y2="38" />
        <line x1="182" y1="66" x2="226" y2="88" />
        <line x1="212" y1="38" x2="226" y2="88" />
        <line x1="140" y1="150" x2="204" y2="132" />
        <line x1="226" y1="88" x2="204" y2="132" />
        <line x1="118" y1="192" x2="210" y2="178" />
        <line x1="204" y1="132" x2="210" y2="178" />
      </g>
      <g fill="var(--accent)" opacity=".55">
        <circle cx="212" cy="38" r="2" />
        <circle cx="226" cy="88" r="2" />
        <circle cx="204" cy="132" r="2" />
        <circle cx="210" cy="178" r="2" />
      </g>

      {/* skeleton */}
      <g stroke="#2b2b28" strokeWidth="2.5" strokeLinecap="round" fill="none">
        <circle cx="124" cy="40" r="11" />                 {/* head */}
        <line x1="120" y1="51" x2="116" y2="62" />          {/* neck */}
        <line x1="116" y1="62" x2="104" y2="118" />         {/* torso */}
        <line x1="116" y1="62" x2="150" y2="72" />          {/* upper arm */}
        <line x1="150" y1="72" x2="182" y2="66" />          {/* forearm */}
        <line x1="104" y1="118" x2="140" y2="150" />        {/* thigh */}
        <line x1="140" y1="150" x2="118" y2="192" />        {/* shin */}
        <line x1="118" y1="192" x2="146" y2="196" />        {/* foot */}
      </g>

      {/* knee angle arc */}
      <circle
        cx="140" cy="150" r="14" fill="none"
        stroke="var(--accent)" strokeWidth="1.5"
        strokeDasharray="25.4 62.6"
        transform="rotate(118 140 150)"
      />

      {/* joints */}
      <g fill="var(--accent)">
        {joints.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="4" />)}
      </g>

      {/* angle chips */}
      <g>
        <line x1="140" y1="150" x2="170" y2="160" stroke="#2b2b28" strokeWidth="1" opacity=".35" />
        <rect x="168" y="150" width="42" height="22" rx="5" fill="#171716" />
        <text x="189" y="161.5" textAnchor="middle" dominantBaseline="central"
              fill="#fff" fontSize="12" fontWeight="800">97°</text>

        <line x1="104" y1="118" x2="76" y2="102" stroke="#2b2b28" strokeWidth="1" opacity=".35" />
        <rect x="32" y="90" width="46" height="22" rx="5" fill="#171716" />
        <text x="55" y="101.5" textAnchor="middle" dominantBaseline="central"
              fill="#fff" fontSize="12" fontWeight="800">118°</text>
      </g>
    </svg>
  );
}
