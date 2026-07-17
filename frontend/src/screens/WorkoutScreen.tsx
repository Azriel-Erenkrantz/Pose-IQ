import PoseFigure from '../components/PoseFigure';
import { useI18n } from '../i18n';

export default function WorkoutScreen() {
  const { t } = useI18n();

  return (
    <div className="max-w-2xl w-full mx-auto px-5 py-7 pb-16">
      {/* Editorial header */}
      <p className="label text-accent mb-2">{t.comingSoon}</p>
      <h1 className="text-4xl font-black text-[#171716] mb-3 leading-tight">{t.workoutTitle}</h1>
      <p className="text-[#6f6e68] text-sm max-w-md leading-relaxed mb-8">{t.workoutPitch}</p>

      {/* Joint-tracking illustration */}
      <div className="panel p-6 mb-8">
        <p className="label mb-4">{t.formTracking}</p>
        <PoseFigure />
      </div>

      {/* Meanwhile: desktop pipeline */}
      <div className="panel p-5 mb-8">
        <p className="text-[#171716] font-bold text-sm mb-2.5">{t.meanwhileTitle}</p>
        <pre dir="ltr" className="bg-[#f7f7f5] border border-[#e6e5e1] rounded-lg px-4 py-3 text-accent text-xs overflow-x-auto text-left num">
          python -m core.pipeline squat &lt;user_id&gt; [weight_kg]
        </pre>
        <p className="text-[#a09f98] text-xs mt-2.5">{t.meanwhileText}</p>
      </div>

      {/* Supported exercises */}
      <p className="label mb-2.5">{t.supportedExercises}</p>
      <div className="panel rows">
        {Object.entries(t.exerciseNames).map(([id, name], i) => (
          <div key={id} className="flex items-center gap-4 px-4 py-3.5">
            <span className="label num w-6">{String(i + 1).padStart(2, '0')}</span>
            <span className="text-[#171716] font-bold text-[15px]">{name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
