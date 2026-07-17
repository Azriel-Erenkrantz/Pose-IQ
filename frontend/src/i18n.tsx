/* eslint-disable react-refresh/only-export-components */
// Lightweight bilingual (en/he) i18n — no external deps.
// `useI18n()` gives { lang, setLang, t, dir }; Hebrew flips the document to RTL.
import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { Equipment, FitnessLevel, TargetGoal, TrainerPersonality } from './api/types';

export type Lang = 'en' | 'he';
const LANG_KEY = 'pose_iq_lang';

interface Dict {
  // Common
  tagline: string;
  loading: string;
  logout: string;
  kg: string;
  bodyweight: string;
  couldNotLoad: string;

  // Nav
  navHome: string;
  navHistory: string;
  navWorkout: string;

  // Login
  email: string;
  password: string;
  loginBtn: string;
  noAccount: string;
  createOne: string;
  fillBothFields: string;
  loginFailed: string;

  // Register
  stepTitles: string[];
  stepOf: (step: number, total: number) => string;
  fitnessLabels: Record<FitnessLevel, string>;
  fitnessDesc: Record<FitnessLevel, string>;
  goalLabels: Record<TargetGoal, string>;
  equipmentLabels: Record<Equipment, string>;
  trainerLabels: Record<TrainerPersonality, string>;
  trainerDesc: Record<TrainerPersonality, string>;
  limitationLabels: Record<string, string>;
  limitationsHint: string;
  namePh: string;
  nameLabel: string;
  passwordPh: string;
  errName: string;
  errEmail: string;
  errEmailValid: string;
  errPassword: string;
  errGoal: string;
  errEquipment: string;
  registerFailed: string;
  back: string;
  next: string;
  createAccount: string;
  goToLogin: string;

  // Dashboard
  greetingMorning: string;
  greetingAfternoon: string;
  greetingEvening: string;
  statSessions: string;
  statRecent: string;
  statAvgScore: string;
  statOutOf: string;
  statExercises: string;
  statTracked: string;
  injuryRisk: string;
  howDoYouFeel: string;
  regionLabels: Record<'upper' | 'core' | 'lower', string>;
  saveHealth: string;
  saving: string;
  savedBang: string;
  recommendedForYou: string;
  noRecsTitle: string;
  noRecsHint: string;
  scenarioHealthy: string;
  scenarioTakeItEasy: string;
  scenarioRecovery: string;
  nextWeight: string;
  yourProgress: string;
  trendLabels: Record<'improving' | 'stable' | 'declining', string>;
  sessionsWord: string;
  repsWord: string;
  weakPrefix: string;
  recentSessions: string;
  viewAll: string;
  logWeight: string;
  save: string;

  // History
  historyTitle: string;
  noSessionsYet: string;
  noSessionsHint: string;
  allSessions: string;

  // Workout
  workoutTitle: string;
  comingSoon: string;
  formTracking: string;
  workoutPitch: string;
  meanwhileTitle: string;
  meanwhileText: string;
  supportedExercises: string;
  exerciseNames: Record<string, string>;
}

const en: Dict = {
  tagline: 'Your AI fitness coach',
  loading: 'Loading…',
  logout: 'Logout',
  kg: 'kg',
  bodyweight: 'Bodyweight',
  couldNotLoad: 'Could not load data. Is the server running?',

  navHome: 'Home',
  navHistory: 'History',
  navWorkout: 'Workout',

  email: 'Email',
  password: 'Password',
  loginBtn: 'Log In',
  noAccount: 'No account?',
  createOne: 'Create one',
  fillBothFields: 'Please fill in both fields.',
  loginFailed: 'Login failed. Check your credentials.',

  stepTitles: ['Create your account', 'Fitness level', 'Training goals',
               'Equipment', 'Coach style', 'Any limitations?'],
  stepOf: (s, t) => `Step ${s} of ${t}`,
  fitnessLabels: { beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced' },
  fitnessDesc: {
    beginner:     'New to fitness or returning after a break',
    intermediate: 'Exercising regularly for 6+ months',
    advanced:     'Training consistently for 2+ years',
  },
  goalLabels: {
    legs: 'Legs', cardio: 'Cardio', abs: 'Abs',
    arms: 'Arms', full_body: 'Full Body', other: 'Other',
  },
  equipmentLabels: { dumbbells: 'Dumbbells', resistance_bands: 'Bands', none: 'No Equipment' },
  trainerLabels: { tough: 'Tough', calm: 'Calm', motivating: 'Motivating' },
  trainerDesc: {
    tough:      'Direct and demanding — short commands, no fluff',
    calm:       'Gentle and patient — smooth, reassuring cues',
    motivating: 'Energetic and encouraging — hypes you up',
  },
  limitationLabels: {
    right_knee: 'Right Knee', left_knee: 'Left Knee', lower_back: 'Lower Back',
    right_shoulder: 'Right Shoulder', left_shoulder: 'Left Shoulder',
    right_elbow: 'Right Elbow', left_elbow: 'Left Elbow',
  },
  limitationsHint: 'Select any joints that need extra care. The app will adapt form checks accordingly. Skip if none apply.',
  namePh: 'Your name',
  nameLabel: 'Name',
  passwordPh: 'At least 6 characters',
  errName: 'Please enter your name.',
  errEmail: 'Please enter your email.',
  errEmailValid: 'Please enter a valid email.',
  errPassword: 'Password must be at least 6 characters.',
  errGoal: 'Pick at least one goal.',
  errEquipment: 'Pick at least one equipment option.',
  registerFailed: 'Registration failed. Please try again.',
  back: '← Back',
  next: 'Next →',
  createAccount: 'Create Account',
  goToLogin: 'Log in',

  greetingMorning: 'Good morning',
  greetingAfternoon: 'Good afternoon',
  greetingEvening: 'Good evening',
  statSessions: 'Sessions',
  statRecent: 'recent',
  statAvgScore: 'Avg Score',
  statOutOf: '/ 100',
  statExercises: 'Exercises',
  statTracked: 'tracked',
  injuryRisk: 'Injury risk detected',
  howDoYouFeel: 'How do you feel today?',
  regionLabels: { upper: 'Upper Body', core: 'Core', lower: 'Lower Body' },
  saveHealth: 'Save & refresh recommendations',
  saving: 'Saving…',
  savedBang: 'Saved!',
  recommendedForYou: 'Recommended for you',
  noRecsTitle: 'No recommendations yet.',
  noRecsHint: 'Rate your health above and tap Save, or make sure you selected goals during registration.',
  scenarioHealthy: 'Healthy',
  scenarioTakeItEasy: 'Take it easy',
  scenarioRecovery: 'Recovery',
  nextWeight: 'Next Working Weight',
  yourProgress: 'Your Progress',
  trendLabels: { improving: 'improving', stable: 'stable', declining: 'declining' },
  sessionsWord: 'sessions',
  repsWord: 'reps',
  weakPrefix: 'Weak',
  recentSessions: 'Recent Sessions',
  viewAll: 'View all →',
  logWeight: '+ log weight',
  save: 'Save',

  historyTitle: 'Training History',
  noSessionsYet: 'No sessions yet.',
  noSessionsHint: 'Complete a workout with the camera pipeline and it will show up here.',
  allSessions: 'All Sessions',

  workoutTitle: 'Live Workout',
  comingSoon: 'In development',
  formTracking: 'Joint tracking · Squat, side view',
  workoutPitch: 'Real-time form correction straight in your browser — camera, pose tracking, rep counting and voice coaching, all on-device.',
  meanwhileTitle: 'Meanwhile, train from the desktop app:',
  meanwhileText: 'Your session is saved automatically and will appear in the dashboard and history.',
  supportedExercises: 'Supported exercises',
  exerciseNames: {
    squat: 'Squat', lunge: 'Lunge',
    biceps_curl: 'Biceps Curl', shoulder_press: 'Shoulder Press',
  },
};

const he: Dict = {
  tagline: 'המאמן האישי החכם שלך',
  loading: 'טוען…',
  logout: 'התנתקות',
  kg: 'ק"ג',
  bodyweight: 'משקל גוף',
  couldNotLoad: 'לא ניתן לטעון נתונים. האם השרת רץ?',

  navHome: 'בית',
  navHistory: 'היסטוריה',
  navWorkout: 'אימון',

  email: 'אימייל',
  password: 'סיסמה',
  loginBtn: 'התחברות',
  noAccount: 'אין חשבון?',
  createOne: 'הרשמה',
  fillBothFields: 'נא למלא את שני השדות.',
  loginFailed: 'ההתחברות נכשלה. בדקו את הפרטים.',

  stepTitles: ['יצירת חשבון', 'רמת כושר', 'מטרות אימון',
               'ציוד', 'סגנון המאמן', 'מגבלות גופניות?'],
  stepOf: (s, t) => `שלב ${s} מתוך ${t}`,
  fitnessLabels: { beginner: 'מתחיל', intermediate: 'בינוני', advanced: 'מתקדם' },
  fitnessDesc: {
    beginner:     'חדש באימונים או חוזר אחרי הפסקה',
    intermediate: 'מתאמן באופן קבוע חצי שנה ומעלה',
    advanced:     'מתאמן ברציפות שנתיים ומעלה',
  },
  goalLabels: {
    legs: 'רגליים', cardio: 'אירובי', abs: 'בטן',
    arms: 'ידיים', full_body: 'כל הגוף', other: 'אחר',
  },
  equipmentLabels: { dumbbells: 'משקולות', resistance_bands: 'גומיות', none: 'בלי ציוד' },
  trainerLabels: { tough: 'קשוח', calm: 'רגוע', motivating: 'מדרבן' },
  trainerDesc: {
    tough:      'ישיר ותובעני — פקודות קצרות, בלי קשקושים',
    calm:       'עדין וסבלני — הנחיות רכות ומרגיעות',
    motivating: 'אנרגטי ומעודד — מקפיץ אותך',
  },
  limitationLabels: {
    right_knee: 'ברך ימין', left_knee: 'ברך שמאל', lower_back: 'גב תחתון',
    right_shoulder: 'כתף ימין', left_shoulder: 'כתף שמאל',
    right_elbow: 'מרפק ימין', left_elbow: 'מרפק שמאל',
  },
  limitationsHint: 'סמנו מפרקים שדורשים זהירות מיוחדת — האפליקציה תתאים את בדיקות הטכניקה. אפשר לדלג אם אין.',
  namePh: 'השם שלך',
  nameLabel: 'שם',
  passwordPh: 'לפחות 6 תווים',
  errName: 'נא להזין שם.',
  errEmail: 'נא להזין אימייל.',
  errEmailValid: 'נא להזין אימייל תקין.',
  errPassword: 'הסיסמה חייבת להכיל לפחות 6 תווים.',
  errGoal: 'בחרו לפחות מטרה אחת.',
  errEquipment: 'בחרו לפחות אפשרות ציוד אחת.',
  registerFailed: 'ההרשמה נכשלה. נסו שוב.',
  back: '→ חזרה',
  next: 'הבא ←',
  createAccount: 'יצירת חשבון',
  goToLogin: 'התחברות',

  greetingMorning: 'בוקר טוב',
  greetingAfternoon: 'צהריים טובים',
  greetingEvening: 'ערב טוב',
  statSessions: 'אימונים',
  statRecent: 'אחרונים',
  statAvgScore: 'ציון ממוצע',
  statOutOf: '/ 100',
  statExercises: 'תרגילים',
  statTracked: 'במעקב',
  injuryRisk: 'זוהה סיכון לפציעה',
  howDoYouFeel: 'איך אתם מרגישים היום?',
  regionLabels: { upper: 'פלג גוף עליון', core: 'ליבה', lower: 'פלג גוף תחתון' },
  saveHealth: 'שמירה ורענון המלצות',
  saving: 'שומר…',
  savedBang: 'נשמר!',
  recommendedForYou: 'מומלץ עבורך',
  noRecsTitle: 'אין המלצות עדיין.',
  noRecsHint: 'דרגו את ההרגשה למעלה ולחצו שמירה, או ודאו שבחרתם מטרות בהרשמה.',
  scenarioHealthy: 'בריא',
  scenarioTakeItEasy: 'בעדינות',
  scenarioRecovery: 'שיקום',
  nextWeight: 'המשקל הבא שלך',
  yourProgress: 'ההתקדמות שלך',
  trendLabels: { improving: 'משתפר', stable: 'יציב', declining: 'יורד' },
  sessionsWord: 'אימונים',
  repsWord: 'חזרות',
  weakPrefix: 'נקודות חולשה',
  recentSessions: 'אימונים אחרונים',
  viewAll: 'לכל האימונים ←',
  logWeight: '+ רישום משקל',
  save: 'שמירה',

  historyTitle: 'היסטוריית אימונים',
  noSessionsYet: 'אין אימונים עדיין.',
  noSessionsHint: 'השלימו אימון מול המצלמה והוא יופיע כאן.',
  allSessions: 'כל האימונים',

  workoutTitle: 'אימון חי',
  comingSoon: 'בפיתוח',
  formTracking: 'מעקב מפרקים · סקוואט, מבט צד',
  workoutPitch: 'תיקון טכניקה בזמן אמת ישירות בדפדפן — מצלמה, זיהוי תנועה, ספירת חזרות ואימון קולי, הכול על המכשיר שלך.',
  meanwhileTitle: 'בינתיים, מתאמנים מאפליקציית המחשב:',
  meanwhileText: 'האימון נשמר אוטומטית ויופיע בדשבורד ובהיסטוריה.',
  supportedExercises: 'תרגילים נתמכים',
  exerciseNames: {
    squat: 'סקוואט', lunge: 'מכרעים', biceps_curl: 'כפיפת מרפקים', shoulder_press: 'לחיצת כתפיים',
  },
};

const DICTS: Record<Lang, Dict> = { en, he };

interface I18nCtx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: Dict;
  dir: 'ltr' | 'rtl';
  locale: string;
  exerciseName: (id: string, fallback: string) => string;
}

const Ctx = createContext<I18nCtx | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() =>
    (localStorage.getItem(LANG_KEY) as Lang) || 'he');

  const dir: 'ltr' | 'rtl' = lang === 'he' ? 'rtl' : 'ltr';

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = dir;
  }, [lang, dir]);

  function setLang(l: Lang) {
    localStorage.setItem(LANG_KEY, l);
    setLangState(l);
  }

  const t = DICTS[lang];
  const value: I18nCtx = {
    lang, setLang, t, dir,
    locale: lang === 'he' ? 'he-IL' : 'en-US',
    exerciseName: (id, fallback) => t.exerciseNames[id] ?? fallback,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18nCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useI18n must be used inside LanguageProvider');
  return ctx;
}
