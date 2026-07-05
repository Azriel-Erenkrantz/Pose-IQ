import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { authApi } from '../api/client';
import {
  AuthToken,
  Equipment,
  FitnessLevel,
  ProfileSetupOptions,
  TargetGoal,
  TrainerPersonality,
} from '../api/types';

interface Props {
  onRegister: (token: AuthToken) => void;
  onGoToLogin: () => void;
}

const TOTAL_STEPS = 6;

const STEP_TITLES = [
  'Create your account',
  'Fitness level',
  'Training goals',
  'Equipment',
  'Coach style',
  'Any limitations?',
];

const FITNESS_LABELS: Record<FitnessLevel, string> = {
  beginner:     'Beginner',
  intermediate: 'Intermediate',
  advanced:     'Advanced',
};

const FITNESS_DESC: Record<FitnessLevel, string> = {
  beginner:     'New to fitness or returning after a break',
  intermediate: 'Exercising regularly for 6+ months',
  advanced:     'Training consistently for 2+ years',
};

const GOAL_LABELS: Record<TargetGoal, string> = {
  legs:      '🦵 Legs',
  cardio:    '❤️ Cardio',
  abs:       '💪 Abs',
  arms:      '🤜 Arms',
  full_body: '🏋️ Full Body',
  other:     '✨ Other',
};

const EQUIPMENT_LABELS: Record<Equipment, string> = {
  dumbbells:        '🏋️ Dumbbells',
  resistance_bands: '🔁 Resistance Bands',
  none:             '🙌 No Equipment',
};

const TRAINER_LABELS: Record<TrainerPersonality, string> = {
  tough:      '🔥 Tough',
  calm:       '🧘 Calm',
  motivating: '⚡ Motivating',
};

const TRAINER_DESC: Record<TrainerPersonality, string> = {
  tough:      'Direct, demanding — short commands, no fluff',
  calm:       'Gentle and patient — smooth, reassuring cues',
  motivating: 'Energetic and encouraging — hypes you up',
};

const LIMITATION_LABELS: Record<string, string> = {
  right_knee:     '🦵 Right Knee',
  left_knee:      '🦵 Left Knee',
  lower_back:     '🔙 Lower Back',
  right_shoulder: '💪 Right Shoulder',
  left_shoulder:  '💪 Left Shoulder',
  right_elbow:    '🤜 Right Elbow',
  left_elbow:     '🤛 Left Elbow',
};

export default function RegisterScreen({ onRegister, onGoToLogin }: Props) {
  const [step, setStep] = useState(1);
  const [options, setOptions] = useState<ProfileSetupOptions | null>(null);

  // Step 1 — account
  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');

  // Step 2 — fitness level
  const [fitnessLevel, setFitnessLevel] = useState<FitnessLevel>('intermediate');

  // Step 3 — goals
  const [goals, setGoals] = useState<TargetGoal[]>([]);

  // Step 4 — equipment
  const [equipment, setEquipment] = useState<Equipment[]>([]);

  // Step 5 — trainer
  const [trainer, setTrainer] = useState<TrainerPersonality>('motivating');

  // Step 6 — limitations
  const [limitations, setLimitations] = useState<string[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  useEffect(() => {
    authApi.options().then(setOptions).catch(() => {});
  }, []);

  function toggleMulti<T>(list: T[], item: T): T[] {
    return list.includes(item) ? list.filter(i => i !== item) : [...list, item];
  }

  function validateStep(): string {
    if (step === 1) {
      if (!name.trim())    return 'Please enter your name.';
      if (!email.trim())   return 'Please enter your email.';
      if (!/\S+@\S+\.\S+/.test(email)) return 'Please enter a valid email.';
      if (password.length < 6) return 'Password must be at least 6 characters.';
    }
    if (step === 3 && goals.length === 0)     return 'Pick at least one goal.';
    if (step === 4 && equipment.length === 0) return 'Pick at least one equipment option.';
    return '';
  }

  function handleNext() {
    const e = validateStep();
    if (e) { setError(e); return; }
    setError('');
    setStep(s => s + 1);
  }

  function handleBack() {
    setError('');
    setStep(s => s - 1);
  }

  async function handleSubmit() {
    setLoading(true);
    setError('');
    try {
      const token = await authApi.register({
        name:                name.trim(),
        email:               email.trim().toLowerCase(),
        password,
        fitness_level:       fitnessLevel,
        trainer_personality: trainer,
        target_goals:        goals,
        equipment,
        limitations,
      });
      onRegister(token);
    } catch (e: any) {
      setError(e.message ?? 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      {/* Progress bar */}
      <View style={styles.progressRow}>
        {Array.from({ length: TOTAL_STEPS }).map((_, i) => (
          <View
            key={i}
            style={[styles.progressDot, i < step && styles.progressDotActive]}
          />
        ))}
      </View>

      <Text style={styles.stepLabel}>Step {step} of {TOTAL_STEPS}</Text>
      <Text style={styles.stepTitle}>{STEP_TITLES[step - 1]}</Text>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {/* Step 1 — Account */}
        {step === 1 && (
          <View style={styles.fieldGroup}>
            <Text style={styles.label}>Name</Text>
            <TextInput
              style={styles.input}
              placeholder="Your name"
              placeholderTextColor="#555"
              value={name}
              onChangeText={setName}
            />
            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="you@example.com"
              placeholderTextColor="#555"
              autoCapitalize="none"
              keyboardType="email-address"
              value={email}
              onChangeText={setEmail}
            />
            <Text style={styles.label}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="At least 6 characters"
              placeholderTextColor="#555"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />
          </View>
        )}

        {/* Step 2 — Fitness level */}
        {step === 2 && (
          <View style={styles.fieldGroup}>
            {(['beginner', 'intermediate', 'advanced'] as FitnessLevel[]).map(level => (
              <Pressable
                key={level}
                style={[styles.radioCard, fitnessLevel === level && styles.radioCardActive]}
                onPress={() => setFitnessLevel(level)}
              >
                <View style={styles.radioRow}>
                  <View style={[styles.radio, fitnessLevel === level && styles.radioSelected]} />
                  <Text style={styles.radioTitle}>{FITNESS_LABELS[level]}</Text>
                </View>
                <Text style={styles.radioDesc}>{FITNESS_DESC[level]}</Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Step 3 — Goals */}
        {step === 3 && (
          <View style={styles.grid}>
            {(options?.target_goals ?? Object.keys(GOAL_LABELS) as TargetGoal[]).map(goal => (
              <Pressable
                key={goal}
                style={[styles.gridCard, goals.includes(goal) && styles.gridCardActive]}
                onPress={() => setGoals(g => toggleMulti(g, goal))}
              >
                <Text style={styles.gridIcon}>{GOAL_LABELS[goal]?.split(' ')[0]}</Text>
                <Text style={styles.gridLabel}>
                  {GOAL_LABELS[goal]?.split(' ').slice(1).join(' ') || goal}
                </Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Step 4 — Equipment */}
        {step === 4 && (
          <View style={styles.grid}>
            {(options?.equipment_options ?? Object.keys(EQUIPMENT_LABELS) as Equipment[]).map(eq => (
              <Pressable
                key={eq}
                style={[styles.gridCard, equipment.includes(eq) && styles.gridCardActive]}
                onPress={() => setEquipment(e => toggleMulti(e, eq))}
              >
                <Text style={styles.gridIcon}>{EQUIPMENT_LABELS[eq]?.split(' ')[0]}</Text>
                <Text style={styles.gridLabel}>
                  {EQUIPMENT_LABELS[eq]?.split(' ').slice(1).join(' ') || eq}
                </Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Step 5 — Trainer personality */}
        {step === 5 && (
          <View style={styles.fieldGroup}>
            {(['tough', 'calm', 'motivating'] as TrainerPersonality[]).map(style => (
              <Pressable
                key={style}
                style={[styles.radioCard, trainer === style && styles.radioCardActive]}
                onPress={() => setTrainer(style)}
              >
                <View style={styles.radioRow}>
                  <View style={[styles.radio, trainer === style && styles.radioSelected]} />
                  <Text style={styles.radioTitle}>{TRAINER_LABELS[style]}</Text>
                </View>
                <Text style={styles.radioDesc}>{TRAINER_DESC[style]}</Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Step 6 — Physical limitations */}
        {step === 6 && (
          <View>
            <Text style={styles.limitationHint}>
              Select any joints or areas that need extra care. The app will adapt form checks accordingly. Skip if none apply.
            </Text>
            <View style={styles.grid}>
              {(options?.limitation_options ?? Object.keys(LIMITATION_LABELS)).map(lim => (
                <Pressable
                  key={lim}
                  style={[styles.gridCard, limitations.includes(lim) && styles.gridCardActive]}
                  onPress={() => setLimitations(l => toggleMulti(l, lim))}
                >
                  <Text style={styles.gridIcon}>{LIMITATION_LABELS[lim]?.split(' ')[0]}</Text>
                  <Text style={styles.gridLabel}>
                    {LIMITATION_LABELS[lim]?.split(' ').slice(1).join(' ') || lim}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        )}

        {error ? <Text style={styles.error}>{error}</Text> : null}
      </ScrollView>

      {/* Navigation buttons */}
      <View style={styles.navRow}>
        {step > 1 ? (
          <Pressable style={styles.backBtn} onPress={handleBack}>
            <Text style={styles.backText}>← Back</Text>
          </Pressable>
        ) : (
          <Pressable style={styles.backBtn} onPress={onGoToLogin}>
            <Text style={styles.backText}>Log in</Text>
          </Pressable>
        )}

        {step < TOTAL_STEPS ? (
          <Pressable style={styles.nextBtn} onPress={handleNext}>
            <Text style={styles.nextText}>Next →</Text>
          </Pressable>
        ) : (
          <Pressable
            style={[styles.nextBtn, loading && styles.btnDisabled]}
            onPress={handleSubmit}
            disabled={loading}
          >
            {loading
              ? <ActivityIndicator color="#0f0f1a" />
              : <Text style={styles.nextText}>Create Account</Text>
            }
          </Pressable>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f1a',
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 32,
  },
  progressRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 20,
  },
  progressDot: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#2a2a40',
  },
  progressDotActive: {
    backgroundColor: '#4ade80',
  },
  stepLabel: { color: '#666', fontSize: 12, marginBottom: 4 },
  stepTitle: { color: '#fff', fontSize: 24, fontWeight: '700', marginBottom: 24 },

  scroll:        { flex: 1 },
  scrollContent: { paddingBottom: 16 },

  fieldGroup: { gap: 8 },
  label: {
    color: '#aaa',
    fontSize: 13,
    fontWeight: '600',
    marginTop: 12,
    marginBottom: 4,
  },
  input: {
    backgroundColor: '#1a1a2e',
    color: '#fff',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#2a2a40',
  },

  radioCard: {
    backgroundColor: '#1a1a2e',
    borderRadius: 14,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2a2a40',
  },
  radioCardActive: { borderColor: '#4ade80' },
  radioRow:  { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 4 },
  radio: {
    width: 18,
    height: 18,
    borderRadius: 9,
    borderWidth: 2,
    borderColor: '#444',
  },
  radioSelected: { borderColor: '#4ade80', backgroundColor: '#4ade80' },
  radioTitle: { color: '#fff', fontSize: 16, fontWeight: '600' },
  radioDesc:  { color: '#888', fontSize: 13, marginLeft: 30 },

  limitationHint: {
    color: '#888',
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 16,
  },

  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  gridCard: {
    width: '30%',
    flexGrow: 1,
    backgroundColor: '#1a1a2e',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a40',
    minHeight: 80,
    justifyContent: 'center',
  },
  gridCardActive: { borderColor: '#4ade80', backgroundColor: '#1a2e1a' },
  gridIcon:  { fontSize: 24, marginBottom: 6 },
  gridLabel: { color: '#ccc', fontSize: 12, textAlign: 'center', fontWeight: '500' },

  error: {
    color: '#ff6b6b',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 12,
  },

  navRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 16,
    gap: 12,
  },
  backBtn: {
    paddingVertical: 14,
    paddingHorizontal: 20,
  },
  backText: { color: '#888', fontSize: 15 },
  nextBtn: {
    flex: 1,
    backgroundColor: '#4ade80',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  btnDisabled: { opacity: 0.6 },
  nextText: { color: '#0f0f1a', fontSize: 16, fontWeight: '700' },
});
