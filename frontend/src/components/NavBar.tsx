import { useI18n } from '../i18n';

export type Tab = 'home' | 'workout' | 'history';

interface Props {
  active: Tab;
  onNavigate: (tab: Tab) => void;
  onLogout: () => void;
}

export default function NavBar({ active, onNavigate, onLogout }: Props) {
  const { t, lang, setLang } = useI18n();

  const tabs: { id: Tab; label: string }[] = [
    { id: 'home',    label: t.navHome },
    { id: 'workout', label: t.navWorkout },
    { id: 'history', label: t.navHistory },
  ];

  return (
    <nav className="sticky top-0 z-20 bg-white border-b border-[#e6e5e1]">
      <div className="max-w-2xl mx-auto px-5 h-14 flex items-center justify-between">
        <span className="font-black text-lg tracking-tight select-none text-[#171716]" dir="ltr">
          POSE-IQ<span className="text-accent">.</span>
        </span>

        <div className="flex items-center h-full">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => onNavigate(tab.id)}
              className={`relative h-full px-3.5 text-sm font-bold transition-colors ${
                active === tab.id ? 'text-[#171716]' : 'text-[#a09f98] hover:text-[#6f6e68]'
              }`}
            >
              {tab.label}
              {active === tab.id && (
                <span className="absolute inset-x-3 bottom-0 h-[2px] bg-[#171716]" />
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setLang(lang === 'he' ? 'en' : 'he')}
            className="btn-ghost text-xs font-bold px-2 py-1"
            title={lang === 'he' ? 'Switch to English' : 'מעבר לעברית'}
          >
            {lang === 'he' ? 'EN' : 'עב'}
          </button>
          <button onClick={onLogout} className="text-xs text-[#a09f98] hover:text-[#171716] transition-colors">
            {t.logout}
          </button>
        </div>
      </div>
    </nav>
  );
}
