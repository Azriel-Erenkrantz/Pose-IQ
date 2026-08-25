/* eslint-disable react-refresh/only-export-components */
// Lightweight bilingual (en/he) i18n — no external deps.
// `useI18n()` gives { lang, setLang, t, dir }; Hebrew flips the document to RTL.
import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { FitnessLevel } from './api/types';

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
  limitationLabels: Record<string, string>;
  limitationsHint: string;
  namePh: string;
  nameLabel: string;
  passwordPh: string;
  errName: string;
  errEmail: string;
  errEmailValid: string;
  errPassword: string;
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
  howDoYouFeel: string;
  regionLabels: Record<'upper' | 'core' | 'lower', string>;
  saveHealth: string;
  saving: string;
  savedBang: string;
  recommendedForYou: string;
  noRecsTitle: string;
  noRecsHint: string;
  rateExercise: string;
  stars: string;
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
  reasonTemplates: Record<string, (p: Record<string, number>) => string>;

  // Live workout
  chooseExercise: string;
  startWorkout: string;
  cameraHint: string;
  notReadyHint: string;
  loadingModel: string;
  cameraError: string;
  modelError: string;
  getIntoPosition: string;
  startingPosition: string;
  endWorkout: string;
  goodForm: string;
  formLabel: string;
  timeLabel: string;
  repFlash: (n: number) => string;
  goFlash: string;
  summaryTitle: string;
  saveSession: string;
  savingSession: string;
  discard: string;
  sessionSaved: string;
  weightUsedLabel: string;
  noRepsYet: string;
  voiceLabel: string;
  adjustStraighten: string;
  adjustBend: string;
  adjustMissing: string;
  jointLabels: Record<string, string>;
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

  stepTitles: ['Create your account', 'Fitness level', 'Any limitations?'],
  stepOf: (s, t) => `Step ${s} of ${t}`,
  fitnessLabels: { beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced' },
  fitnessDesc: {
    beginner:     'New to fitness or returning after a break',
    intermediate: 'Exercising regularly for 6+ months',
    advanced:     'Training consistently for 2+ years',
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
  howDoYouFeel: 'How do you feel today?',
  regionLabels: { upper: 'Upper Body', core: 'Core', lower: 'Lower Body' },
  saveHealth: 'Save & refresh recommendations',
  saving: 'Saving…',
  savedBang: 'Saved!',
  recommendedForYou: 'Recommended for you',
  noRecsTitle: 'No recommendations yet.',
  noRecsHint: 'Rate your health above and tap Save, or make sure you selected goals during registration.',
  rateExercise: 'Rate this exercise',
  stars: 'stars',
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
    push_up: 'Push-up', tricep_dip: 'Tricep Dip', plank: 'Plank',
    crunch: 'Crunch', russian_twist: 'Russian Twist', leg_raise: 'Leg Raise',
    deadlift: 'Deadlift', calf_raise: 'Calf Raise', glute_bridge: 'Glute Bridge',
  },
  reasonTemplates: {
    rated_by_you: p => `You rated this ${p.rating.toFixed(0)}/5`,
    predicted_from_ratings: () => 'Predicted from your other ratings',
    no_ratings_yet: () => "New to you — based on other users' ratings",
    no_weight_logged: () => 'No weight logged yet — start with bodyweight or a light load and log it after each session.',
    form_dropped: p => `Form score dropped to ${p.score.toFixed(0)} at ${num(p.weight)} kg — reduce the load and rebuild clean technique.`,
    ready_to_increase: p => `${p.clean} sessions at ${num(p.weight)} kg with form ≥ ${p.threshold.toFixed(0)} — ready to add ${num(p.increment)} kg.`,
    stay_here: p => `${p.clean}/${p.needed} clean sessions at ${num(p.weight)} kg (form ≥ ${p.threshold.toFixed(0)}) — stay here until the technique is consistent.`,
  },

  chooseExercise: 'Choose an exercise',
  startWorkout: 'Start workout',
  cameraHint: 'Allow camera access when prompted. Step back so your whole body is in frame — a side view works best.',
  notReadyHint: 'Angle ranges missing — run the trainer on the server first',
  loadingModel: 'Loading pose model…',
  cameraError: 'Camera blocked. Click the camera icon in the address bar, allow access, and try again.',
  modelError: 'Could not load the pose model — check your internet connection and try again.',
  getIntoPosition: 'Get into starting position',
  startingPosition: 'Starting position',
  endWorkout: 'End workout',
  goodForm: 'Good form — keep it up',
  formLabel: 'Form',
  timeLabel: 'Time',
  repFlash: n => `REP ${n}`,
  goFlash: 'GO!',
  summaryTitle: 'Workout summary',
  saveSession: 'Save session',
  savingSession: 'Saving…',
  discard: 'Discard',
  sessionSaved: 'Saved to your history',
  weightUsedLabel: 'Weight used (kg, optional)',
  noRepsYet: 'No completed reps — nothing to save.',
  voiceLabel: 'Voice',
  adjustStraighten: 'straighten',
  adjustBend: 'bend more',
  adjustMissing: "can't see it — step back or adjust the camera",
  jointLabels: {
    right_knee: 'Right knee', left_knee: 'Left knee',
    right_hip: 'Right hip', left_hip: 'Left hip',
    right_elbow: 'Right elbow', left_elbow: 'Left elbow',
    right_shoulder: 'Right shoulder', left_shoulder: 'Left shoulder',
    right_ankle: 'Right ankle', left_ankle: 'Left ankle',
    spine: 'Back',
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

  stepTitles: ['יצירת חשבון', 'רמת כושר', 'מגבלות גופניות?'],
  stepOf: (s, t) => `שלב ${s} מתוך ${t}`,
  fitnessLabels: { beginner: 'מתחיל', intermediate: 'בינוני', advanced: 'מתקדם' },
  fitnessDesc: {
    beginner:     'חדש באימונים או חוזר אחרי הפסקה',
    intermediate: 'מתאמן באופן קבוע חצי שנה ומעלה',
    advanced:     'מתאמן ברציפות שנתיים ומעלה',
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
  howDoYouFeel: 'איך אתם מרגישים היום?',
  regionLabels: { upper: 'פלג גוף עליון', core: 'ליבה', lower: 'פלג גוף תחתון' },
  saveHealth: 'שמירה ורענון המלצות',
  saving: 'שומר…',
  savedBang: 'נשמר!',
  recommendedForYou: 'מומלץ עבורך',
  noRecsTitle: 'אין המלצות עדיין.',
  noRecsHint: 'דרגו את ההרגשה למעלה ולחצו שמירה, או ודאו שבחרתם מטרות בהרשמה.',
  rateExercise: 'דרגו תרגיל זה',
  stars: 'כוכבים',
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
    push_up: 'שכיבות סמיכה', tricep_dip: 'דיפס לטריצפס', plank: 'פלאנק',
    crunch: 'כפיפות בטן', russian_twist: 'פיתול רוסי', leg_raise: 'הרמות רגליים',
    deadlift: 'הרמת מתים', calf_raise: 'הרמת עקבים', glute_bridge: 'גשר ישבן',
  },
  reasonTemplates: {
    rated_by_you: p => `דירגתם זאת ${p.rating.toFixed(0)}/5`,
    predicted_from_ratings: () => 'חיזוי לפי הדירוגים האחרים שלכם',
    no_ratings_yet: () => 'חדש עבורכם — מבוסס על דירוגי משתמשים אחרים',
    no_weight_logged: () => 'עדיין לא נרשם משקל — התחילו עם משקל גוף או עומס קל ותעדו אותו אחרי כל אימון.',
    form_dropped: p => `ציון הטכניקה ירד ל-${p.score.toFixed(0)} במשקל ${num(p.weight)} ק"ג — הפחיתו את העומס ובנו מחדש טכניקה נקייה.`,
    ready_to_increase: p => `${p.clean} אימונים במשקל ${num(p.weight)} ק"ג עם טכניקה ≥ ${p.threshold.toFixed(0)} — מוכנים להוסיף ${num(p.increment)} ק"ג.`,
    stay_here: p => `${p.clean}/${p.needed} אימונים נקיים במשקל ${num(p.weight)} ק"ג (טכניקה ≥ ${p.threshold.toFixed(0)}) — הישארו כאן עד שהטכניקה תתייצב.`,
  },

  chooseExercise: 'בחרו תרגיל',
  startWorkout: 'התחלת אימון',
  cameraHint: 'אשרו גישה למצלמה כשתתבקשו. התרחקו כך שכל הגוף בפריים — מבט צד עובד הכי טוב.',
  notReadyHint: 'חסרים טווחי זוויות — יש להריץ קודם את המאמן בשרת',
  loadingModel: 'טוען מודל זיהוי תנועה…',
  cameraError: 'המצלמה חסומה. לחצו על אייקון המצלמה בשורת הכתובת, אשרו גישה ונסו שוב.',
  modelError: 'טעינת מודל זיהוי התנועה נכשלה — בדקו חיבור אינטרנט ונסו שוב.',
  getIntoPosition: 'היכנסו לעמדת הפתיחה',
  startingPosition: 'עמדת פתיחה',
  endWorkout: 'סיום אימון',
  goodForm: 'טכניקה טובה — המשיכו כך',
  formLabel: 'טכניקה',
  timeLabel: 'זמן',
  repFlash: n => `חזרה ${n}`,
  goFlash: 'קדימה!',
  summaryTitle: 'סיכום אימון',
  saveSession: 'שמירת אימון',
  savingSession: 'שומר…',
  discard: 'ביטול',
  sessionSaved: 'נשמר בהיסטוריה שלך',
  weightUsedLabel: 'משקל בשימוש (ק"ג, לא חובה)',
  noRepsYet: 'לא הושלמו חזרות — אין מה לשמור.',
  voiceLabel: 'קול',
  adjustStraighten: 'ליישר',
  adjustBend: 'לכופף יותר',
  adjustMissing: 'לא רואים — התרחקו או כווננו את המצלמה',
  jointLabels: {
    right_knee: 'ברך ימין', left_knee: 'ברך שמאל',
    right_hip: 'ירך ימין', left_hip: 'ירך שמאל',
    right_elbow: 'מרפק ימין', left_elbow: 'מרפק שמאל',
    right_shoulder: 'כתף ימין', left_shoulder: 'כתף שמאל',
    right_ankle: 'קרסול ימין', left_ankle: 'קרסול שמאל',
    spine: 'גב',
  },
};

// Matches Python's ':g' formatting used server-side for reason_params.
function num(x: number): string { return x.toString(); }

const DICTS: Record<Lang, Dict> = { en, he };

interface I18nCtx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: Dict;
  dir: 'ltr' | 'rtl';
  locale: string;
  exerciseName: (id: string, fallback: string) => string;
  formatReason: (code: string, params: Record<string, number>, fallback: string) => string;
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
    formatReason: (code, params, fallback) => t.reasonTemplates[code]?.(params) ?? fallback,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18nCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useI18n must be used inside LanguageProvider');
  return ctx;
}
