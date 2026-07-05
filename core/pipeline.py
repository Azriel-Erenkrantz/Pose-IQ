import cv2
import time
import logging
from detection.camera_stream import CameraStream
from detection.pose_detector import PoseDetector
from detection.angle_calculator import AngleCalculator
from exercise.posture_rules import PostureRules
from exercise.exercise_model import ExerciseModel
from exercise.exercise_state_machine import ExerciseStateMachine
from user.workout_history import WorkoutHistory, new_session, rep_form_score, RepRecord, JOINT_TO_MUSCLE
from coaching.voice_coach import VoiceCoach
from ml.data_collector import DataCollector
from ml.predictor import FormPredictor
from app_model import User
# Recommendation engine integration: see recommendation/bridge.py when ready to wire up.

logging.basicConfig(level=logging.INFO, format='%(message)s')

_JOINT_LABELS = {
    'right_knee': 'Right knee', 'left_knee': 'Left knee',
    'spine': 'Back / spine', 'legs_spread': 'Feet width',
    'right_elbow': 'Right elbow', 'left_elbow': 'Left elbow',
    'right_arm_body': 'Right arm', 'left_arm_body': 'Left arm',
}

_READY_HINTS = {
    'right_knee':    {'too low': 'straighten right leg',  'too high': 'bend right knee more'},
    'left_knee':     {'too low': 'straighten left leg',   'too high': 'bend left knee more'},
    'spine':         {'too low': 'straighten your back',  'too high': 'straighten your back'},
    'legs_spread':   {'too low': 'spread feet wider',     'too high': 'bring feet closer'},
    'right_elbow':   {'too low': 'open right elbow',      'too high': 'bend right elbow to 90°'},
    'left_elbow':    {'too low': 'open left elbow',       'too high': 'bend left elbow to 90°'},
    'right_arm_body':{'too low': 'raise right arm',       'too high': 'lower right arm'},
    'left_arm_body': {'too low': 'raise left arm',        'too high': 'lower left arm'},
}

class PosePipeline:
    def __init__(self, user: User, exercise_id: str = None):
        self.user = user
        logging.info(f"Welcome back, {user.name}! (Level: {user.fitness_level.value})")

        logging.info("Initializing Real-Time 3D Pipeline...")
        self.camera = CameraStream()
        self.detector = PoseDetector()
        self.exercise_model = ExerciseModel()

        if exercise_id is None:
            exercise_id = self._select_exercise()

        exercise = self.exercise_model.get_exercise(exercise_id)
        if not exercise:
            raise ValueError(f"Exercise '{exercise_id}' not found. Available: {self.exercise_model.list_exercises()}")
        self.exercise = exercise
        self.state_machine = ExerciseStateMachine(exercise, self.exercise_model)
        self.rules = PostureRules(exercise, user.threshold_modifier, user.limited_joints)
        logging.info(f"Exercise: {exercise.name}")

        self.history = WorkoutHistory()
        self.session = new_session(exercise.id, exercise.name)
        # PI-46/81: collect per-frame angles, tagged with a stable user_id, for ML training.
        self.collector = DataCollector(user_id=user.user_id)
        # PI-87: live ML heads. Each loads its saved model or no-ops (available=False)
        # if it hasn't been trained yet (e.g. plank). Quality is one-class, per exercise.
        # predict_every throttles inference to keep the loop responsive.
        self.ml_exercise = FormPredictor('exercise', predict_every=5)
        self.ml_phase = FormPredictor('phase', predict_every=5)
        self.ml_quality = FormPredictor(f'quality_{exercise.id}', predict_every=5)
        self._ml_pred: dict = {}
        self._ml_last_form = None
        self.coach = VoiceCoach(style=user.trainer_personality.value)
        self.coach.on_welcome(user.name, exercise.name)
        self._current_rep_errors: set = set()
        self._rep_flash_until: float = 0.0
        self._go_flash_until: float = 0.0
        self._was_started: bool = False
        self._show_debug: bool = False

        self.SKELETON_CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

        self.ERROR_TO_LINES = {
            'spine': [(11, 23), (12, 24), (23, 24)],
            'right_knee': [(24, 26), (26, 28)],
            'left_knee': [(23, 25), (25, 27)],
            'right_arm_body': [(12, 14), (14, 16)],
            'left_arm_body': [(11, 13), (13, 15)],
            'right_elbow': [(12, 14), (14, 16)],
            'left_elbow': [(11, 13), (13, 15)],
            'legs_spread': [(23, 25), (24, 26), (23, 24)]
        }

    def _select_exercise(self) -> str:
        exercises = self.exercise_model.list_exercises()
        if len(exercises) == 1:
            return exercises[0]

        print("\nSelect exercise:")
        for i, ex_id in enumerate(exercises, 1):
            ex = self.exercise_model.get_exercise(ex_id)
            print(f"  {i}. {ex.name if ex else ex_id}")

        while True:
            choice = input("Choose: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(exercises):
                return exercises[int(choice) - 1]
            print(f"Please enter 1-{len(exercises)}")

    # ── joint index → severity color for landmark dots ──────────────────
    _JOINT_LANDMARK_IDX = {
        'spine': [23, 24], 'right_knee': [26], 'left_knee': [25],
        'right_elbow': [14], 'left_elbow': [13],
        'right_arm_body': [12], 'left_arm_body': [11],
        'legs_spread': [23, 24],
    }
    _SEV_COLOR = {
        'high':   (60,  60, 255),
        'medium': (40, 165, 255),
        'low':    (0,  220, 200),
    }

    def _draw_panel(self, frame, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.60):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_skeleton(self, frame, landmarks, posture_errors):
        error_severity = {}
        red_lines = set()
        for error in (posture_errors or []):
            jn = error['joint']
            error_severity[jn] = error['severity']
            for line in self.ERROR_TO_LINES.get(jn, []):
                red_lines.add(line)

        for s, e in self.SKELETON_CONNECTIONS:
            if s in landmarks and e in landmarks:
                p1 = (int(landmarks[s].x), int(landmarks[s].y))
                p2 = (int(landmarks[e].x), int(landmarks[e].y))
                color = (50, 50, 255) if (s, e) in red_lines else (0, 220, 100)
                cv2.line(frame, p1, p2, color, 3)

        dot_colors: dict = {}
        for joint, sev in error_severity.items():
            c = self._SEV_COLOR.get(sev, (255, 255, 255))
            for idx in self._JOINT_LANDMARK_IDX.get(joint, []):
                dot_colors[idx] = c

        for idx, pt in landmarks.items():
            if idx > 10:
                c = dot_colors.get(idx, (240, 240, 240))
                r = 8 if idx in dot_colors else 5
                cv2.circle(frame, (int(pt.x), int(pt.y)), r, c, -1)
                cv2.circle(frame, (int(pt.x), int(pt.y)), r, (0, 0, 0), 1)

    def draw_hud(self, frame, sm_result, posture_issues, elapsed_seconds):
        h, w = frame.shape[:2]

        ex_name    = sm_result['exercise']
        phase_name = sm_result['phase']
        reps       = sm_result['rep_count']
        started    = sm_result['started']
        ph_idx     = sm_result.get('phase_index', 0)
        ph_count   = sm_result.get('phase_count', 1)

        # ── top panel ────────────────────────────────────────────────────
        self._draw_panel(frame, 0, 0, w, 92)

        cv2.putText(frame, ex_name, (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

        if started:
            instruction  = sm_result.get('instruction', '')
            phase_text   = f"{phase_name}  ({ph_idx + 1}/{ph_count})  —  {instruction}" if instruction else f"Phase: {phase_name}  ({ph_idx + 1}/{ph_count})"
            phase_color  = (0, 210, 120)
        else:
            phase_text  = sm_result.get('instruction', "Get into starting position")
            phase_color = (200, 180, 60)
        cv2.putText(frame, phase_text, (14, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, phase_color, 1)

        mins, secs = divmod(int(elapsed_seconds), 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 105, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

        rep_color = (0, 230, 80) if reps > 0 else (160, 160, 160)
        cv2.putText(frame, f"Reps: {reps}", (w - 115, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, rep_color, 2)

        # ── phase progress bar ────────────────────────────────────────────
        bar_y = 92
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + 6), (35, 35, 35), -1)
        if started and ph_count > 0:
            fill = int(w * (ph_idx + 1) / ph_count)
            cv2.rectangle(frame, (0, bar_y), (fill, bar_y + 6), (0, 190, 100), -1)

        # ── readiness checklist (before start) ───────────────────────────
        readiness = sm_result.get('readiness', {})
        if not started and readiness:
            num_ok   = sum(1 for v in readiness.values() if v is None)
            num_all  = len(readiness)
            row_h    = 28
            card_h   = num_all * row_h + 46
            card_y0  = h - 52 - card_h
            self._draw_panel(frame, 0, card_y0, 440, h - 52)

            cv2.putText(frame, f"Starting position  {num_ok}/{num_all}", (14, card_y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

            y = card_y0 + 44
            for joint, status in readiness.items():
                label = _JOINT_LABELS.get(joint, joint)
                if status is None:
                    # ✓ done
                    cv2.circle(frame, (14, y - 5), 7, (0, 200, 80), -1)
                    cv2.circle(frame, (14, y - 5), 7, (0, 0, 0), 1)
                    cv2.putText(frame, label, (28, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (140, 220, 140), 1)
                else:
                    hint = _READY_HINTS.get(joint, {}).get(status, status)
                    cv2.circle(frame, (14, y - 5), 7, (0, 140, 255), -1)
                    cv2.circle(frame, (14, y - 5), 7, (0, 0, 0), 1)
                    cv2.putText(frame, f"{label}  →  {hint}", (28, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 200, 100), 1)
                y += row_h

        # ── error cards (bottom-left, during exercise) ────────────────────
        elif posture_issues:
            card_h = len(posture_issues) * 32 + 14
            card_y0 = h - 52 - card_h
            self._draw_panel(frame, 0, card_y0, 430, h - 52)
            y = card_y0 + 24
            for issue in posture_issues:
                c = self._SEV_COLOR.get(issue['severity'], self._SEV_COLOR['high'])
                cv2.circle(frame, (16, y - 6), 6, c, -1)
                cv2.putText(frame, issue['message'], (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.57, (235, 235, 235), 1)
                y += 32

        # ── bottom status bar ─────────────────────────────────────────────
        self._draw_panel(frame, 0, h - 46, w, h)

        live_score = max(0, 100 - len(self._current_rep_errors) * 15)
        sc = (0, 210, 80) if live_score >= 80 else (0, 165, 255) if live_score >= 60 else (50, 50, 255)
        cv2.putText(frame, f"Form: {live_score:.0f}/100", (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, sc, 2)

        badge = f"{self.user.name}  |  {self.user.fitness_level.value}"
        cv2.putText(frame, badge, (w - 270, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)

        # ── GO flash (centre screen, triggered once on start) ────────────
        now = time.time()
        if now < self._go_flash_until:
            fade = min(1.0, (self._go_flash_until - now) / 0.4)
            g    = int(230 * fade)
            text = "GO!"
            sz   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 5)[0]
            tx   = (w - sz[0]) // 2
            ty   = h // 2 + sz[1] // 2
            cv2.putText(frame, text, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, g, g), 5)

        # ── rep flash (centre screen) ─────────────────────────────────────
        if now < self._rep_flash_until:
            fade = min(1.0, (self._rep_flash_until - now) / 0.5)
            g    = int(220 * fade)
            text = f"REP  {reps}!"
            sz   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 4)[0]
            tx   = (w - sz[0]) // 2
            ty   = h // 2 + sz[1] // 2
            cv2.putText(frame, text, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 7)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, g, 0), 4)

    def draw_debug_angles(self, frame, landmarks, angles):
        mapping = {
            'right_knee': 26, 'left_knee': 25,
            'right_elbow': 14, 'left_elbow': 13,
            'right_arm_body': 12, 'left_arm_body': 11,
            'spine': 24, 'legs_spread': 23,
        }
        for joint, angle in angles.items():
            if joint in mapping and mapping[joint] in landmarks:
                pt = landmarks[mapping[joint]]
                cv2.putText(frame, f"{joint[:5]}:{int(angle)}", (int(pt.x) + 10, int(pt.y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 220), 1)

    def draw_ml_debug(self, frame):
        """PI-87: experimental ML signals overlay (toggle with 'd'). Kept out of the
        main HUD on purpose — the quality head is one-class on un-trimmed 'good' data,
        so it's a developer signal until the training set is cleaned."""
        preds = self._ml_pred
        if not preds:
            return

        rows = []
        ex = preds.get('exercise')
        if ex:
            rows.append((f"ex: {ex['label']} {ex['confidence']:.0%}", (190, 220, 190)))
        ph = preds.get('phase')
        if ph:
            rows.append((f"phase: {ph['label']} {ph['confidence']:.0%}", (190, 220, 190)))
        q = preds.get('quality')
        if q:
            c = (0, 210, 120) if q['label'] == 'good' else (60, 120, 255)
            rows.append((f"form: {q['label']} ({q['score']:+.2f})", c))
        if not rows:
            return

        h, w = frame.shape[:2]
        pad, rh = 10, 22
        bw, bh = 250, 24 + len(rows) * rh
        x0, y0 = w - bw - 8, 100
        self._draw_panel(frame, x0, y0, x0 + bw, y0 + bh)
        cv2.putText(frame, "ML (experimental)", (x0 + pad, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        y = y0 + 18 + rh
        for text, c in rows:
            cv2.putText(frame, text, (x0 + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)
            y += rh

    def run(self):
        logging.info("Starting pipeline loop. Press 'q' to exit.")
        session_start = time.time()

        while True:
            success, frame = self.camera.read_frame()
            if not success:
                break

            landmarks = self.detector.find_pose(frame)
            angles = AngleCalculator.get_body_angles(landmarks)

            sm_result = self.state_machine.update(angles if angles else {})

            posture_issues = []
            if landmarks and angles and sm_result['started']:
                current_phase = self.state_machine.current_phase
                # Merge global safety constraints with the current phase rules (mirrors
                # ExerciseStateMachine._get_active_rules) — analyze() expects the merged dict.
                active_rules = {**self.exercise.global_constraints, **current_phase.angles}
                posture_issues = self.rules.analyze(angles, active_rules)

                # PI-46/81: snapshot this frame for ML training (collector samples internally).
                self.collector.collect(
                    exercise_id=self.exercise.id,
                    rep_number=sm_result['rep_count'],
                    phase_name=sm_result['phase'],
                    phase_index=sm_result['phase_index'],
                    angles=angles,
                    in_phase=(len(posture_issues) == 0),
                    quality_score=max(0, 100 - len(posture_issues) * 15),
                )

                # PI-87: feed the live ML heads (each keeps its own rolling buffer).
                self._ml_pred = {
                    'exercise': self.ml_exercise.update(angles),
                    'phase': self.ml_phase.update(angles),
                    'quality': self.ml_quality.update(angles),
                }
                q = self._ml_pred.get('quality')
                if q and q['label'] != self._ml_last_form:
                    self._ml_last_form = q['label']
                    logging.info(f"[ML] form -> {q['label']} (score {q['score']:+.2f})")

                self.draw_skeleton(frame, landmarks, posture_issues)
                self.draw_debug_angles(frame, landmarks, angles)

                for issue in posture_issues:
                    self._current_rep_errors.add(issue['joint'])
                    if issue['severity'] == 'high':
                        self.coach.on_error(issue['joint'])

                if sm_result['transitioned']:
                    logging.info(f">> Phase: {sm_result['phase']}")

                if sm_result['started'] and not self._was_started:
                    self._was_started = True
                    self._go_flash_until = time.time() + 1.2
                    self.coach.on_start()

                if sm_result['transitioned']:
                    self.coach.on_phase_transition(
                        sm_result['phase'],
                        sm_result.get('instruction', '')
                    )

                if sm_result['completed_rep']:
                    rep_num = sm_result['rep_count']
                    score = rep_form_score(list(self._current_rep_errors))
                    self.session.rep_records.append(RepRecord(
                        rep_number=rep_num,
                        error_joints=list(self._current_rep_errors),
                        form_score=score
                    ))
                    self.session.total_reps = rep_num
                    self._current_rep_errors = set()
                    self._rep_flash_until = time.time() + 1.5
                    self.coach.on_rep_complete(score)
                    logging.info(f"Rep {rep_num} complete! Form score: {score:.0f}/100")
            elif landmarks and angles:
                self.draw_skeleton(frame, landmarks, [])

            if self._show_debug and landmarks and angles:
                self.draw_debug_angles(frame, landmarks, angles)
                self.draw_ml_debug(frame)

            self.draw_hud(frame, sm_result, posture_issues, time.time() - session_start)

            cv2.imshow('Pose-IQ 3D Feedback', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                self._show_debug = not self._show_debug

        self.session.duration_seconds = round(time.time() - session_start, 1)
        self.session.total_reps = self.state_machine.rep_count
        self.collector.finish()          # PI-46/81: flush remaining buffered rows
        self.camera.cap.release()
        cv2.destroyAllWindows()

        if self.session.total_reps > 0:
            self.coach.on_session_end()
            self.history.save_session(self.session)
            self._print_summary()
        self.coach.shutdown()

    def _print_summary(self):
        s = self.session
        mins, secs = divmod(int(s.duration_seconds), 60)
        print("\n" + "=" * 50)
        print(f"  SESSION SUMMARY — {s.exercise_name}")
        print("=" * 50)
        print(f"  Date     : {s.date[:10]}  {s.date[11:16]}")
        print(f"  Duration : {mins}:{secs:02d}")
        print(f"  Reps     : {s.total_reps}")
        print(f"  Score    : {s.overall_score:.0f}/100")
        print()

        if s.rep_records:
            print("  Rep breakdown:")
            for rep in s.rep_records:
                if rep.error_joints:
                    joints_str = ', '.join(rep.error_joints)
                    print(f"    Rep {rep.rep_number:>2}: {rep.form_score:.0f}/100  ✗ {joints_str}")
                else:
                    print(f"    Rep {rep.rep_number:>2}: {rep.form_score:.0f}/100  ✓ Perfect")
            print()

        metrics = self.history.get_progress_metrics(s.exercise_id)
        if metrics.get('total_sessions', 0) > 1:
            print(f"  Progress  ({metrics['total_sessions']} sessions):")
            print(f"    Avg score (last 5): {metrics['avg_score_recent']}")
            trend = metrics['score_trend']
            trend_icon = {'improving': '↑', 'declining': '↓', 'stable': '→', 'neutral': '–'}.get(trend, '')
            print(f"    Trend: {trend} {trend_icon}")
            if metrics['weak_joints']:
                print("    Weak areas:")
                for joint, count in metrics['weak_joints']:
                    muscle = JOINT_TO_MUSCLE.get(joint, joint)
                    print(f"      • {muscle} ({count} errors)")

        print("=" * 50 + "\n")

if __name__ == "__main__":
    import sys
    from user import service as user_service

    exercise_id = sys.argv[1] if len(sys.argv) > 1 else None
    user_id     = sys.argv[2] if len(sys.argv) > 2 else None

    if not user_id:
        logging.error("Usage: python -m core.pipeline [exercise_id] <user_id>")
        raise SystemExit(1)

    _user = user_service.get_user(user_id)
    if not _user:
        logging.error(f"User '{user_id}' not found. Create an account via the Pose-IQ app.")
        raise SystemExit(1)

    PosePipeline(user=_user, exercise_id=exercise_id).run()
