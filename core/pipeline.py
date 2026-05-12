import cv2
import time
import logging
from camera_stream import CameraStream
from pose_detector import PoseDetector
from angle_calculator import AngleCalculator
from posture_rules import PostureRules
from exercise_model import ExerciseModel
from exercise_state_machine import ExerciseStateMachine
from user_profile import UserProfile
from onboarding import run_onboarding

logging.basicConfig(level=logging.INFO, format='%(message)s')

class PosePipeline:
    def __init__(self, exercise_id: str = None):
        if UserProfile.exists():
            self.profile = UserProfile.load()
            logging.info(f"Welcome back, {self.profile.name}! (Level: {self.profile.fitness_level})")
        else:
            self.profile = run_onboarding()

        logging.info("Initializing Real-Time 3D Pipeline...")
        self.camera = CameraStream()
        self.detector = PoseDetector()
        self.exercise_model = ExerciseModel()

        if exercise_id is None:
            exercise_id = self._select_exercise()

        exercise = self.exercise_model.get_exercise(exercise_id)
        if not exercise:
            raise ValueError(f"Exercise '{exercise_id}' not found. Available: {self.exercise_model.list_exercises()}")
        self.state_machine = ExerciseStateMachine(exercise)
        self.rules = PostureRules(exercise, self.profile)
        logging.info(f"Exercise: {exercise.name}")

    def _select_exercise(self) -> str:
        exercises = self.profile.preferred_exercises
        if len(exercises) == 1:
            return exercises[0]
        print("\nSelect exercise:")
        for i, ex_id in enumerate(exercises, 1):
            ex = self.exercise_model.get_exercise(ex_id)
            print(f"  {i}. {ex.name}")
        while True:
            choice = input("Choose: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(exercises):
                return exercises[int(choice) - 1]
            print(f"Please enter 1-{len(exercises)}")

        self.current_state = None
        self.last_errors = []
        self.ready_counter = 0
        self.FRAMES_TO_READY = 10

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

    def draw_skeleton(self, frame, landmarks, posture_errors):
        red_lines = set()
        if posture_errors:
            for error in posture_errors:
                joint_name = error['joint']
                if joint_name in self.ERROR_TO_LINES:
                    for line in self.ERROR_TO_LINES[joint_name]:
                        red_lines.add(line)

        for start_idx, end_idx in self.SKELETON_CONNECTIONS:
            if start_idx in landmarks and end_idx in landmarks:
                p1 = (int(landmarks[start_idx].x), int(landmarks[start_idx].y))
                p2 = (int(landmarks[end_idx].x), int(landmarks[end_idx].y))
                
                color = (0, 0, 255) if (start_idx, end_idx) in red_lines else (0, 255, 0)
                cv2.line(frame, p1, p2, color, 3) 

        for idx, pt in landmarks.items():
            if idx > 10: 
                cv2.circle(frame, (int(pt.x), int(pt.y)), 4, (255, 255, 255), -1)

    def draw_exercise_info(self, frame, sm_result, posture_issues=None):
        h, w = frame.shape[:2]

        exercise_name = sm_result['exercise']
        phase = sm_result['phase']
        reps = sm_result['rep_count']
        started = sm_result['started']

        if self.profile:
            cv2.putText(frame, f"{self.profile.name} ({self.profile.fitness_level})", (w - 350, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.putText(frame, exercise_name, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        phase_color = (0, 255, 255) if started else (128, 128, 128)
        phase_text = f"Phase: {phase}" if started else "Get into starting position"
        cv2.putText(frame, phase_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

        cv2.putText(frame, f"Reps: {reps}", (w - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if sm_result.get('completed_rep'):
            cv2.putText(frame, "REP!", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        if posture_issues:
            y_offset = 120
            for issue in posture_issues:
                severity_color = {
                    'high': (0, 0, 255),
                    'medium': (0, 165, 255),
                    'low': (0, 255, 255)
                }.get(issue['severity'], (0, 0, 255))
                cv2.putText(frame, issue['message'], (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, severity_color, 2)
                y_offset += 30

    def draw_debug_angles(self, frame, landmarks, angles):
        mapping = {
            'right_knee': 26, 'left_knee': 25,
            'right_elbow': 14, 'left_elbow': 13,
            'right_arm_body': 12, 'left_arm_body': 11,
            'spine': 24,
            'legs_spread': 23
        }
        for joint, angle in angles.items():
            if joint in mapping and mapping[joint] in landmarks:
                pt = landmarks[mapping[joint]]
                # מצייר את הזווית בטקסט צהוב ליד המפרק
                cv2.putText(frame, f"{joint}: {int(angle)}", (int(pt.x) + 10, int(pt.y) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def run(self):
        logging.info("Starting pipeline loop. Press 'q' to exit.")
        
        while True:
            start_time = time.time()
            success, frame = self.camera.read_frame()
            if not success: break

            landmarks = self.detector.find_pose(frame)
            angles = AngleCalculator.get_body_angles(landmarks)

            sm_result = self.state_machine.update(angles if angles else {})

            posture_issues = []
            if landmarks and angles and sm_result['started']:
                current_phase = self.state_machine.current_phase
                posture_issues = self.rules.analyze(angles, current_phase)
                self.draw_skeleton(frame, landmarks, posture_issues)
                self.draw_debug_angles(frame, landmarks, angles)

                if sm_result['transitioned']:
                    logging.info(f">> Phase: {sm_result['phase']}")
                if sm_result['completed_rep']:
                    logging.info(f"Rep {sm_result['rep_count']} complete!")
            elif landmarks and angles:
                self.draw_skeleton(frame, landmarks, [])
                self.draw_debug_angles(frame, landmarks, angles)

            self.draw_exercise_info(frame, sm_result, posture_issues)

            cv2.imshow('Pose-IQ 3D Feedback', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.camera.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    exercise_id = sys.argv[1] if len(sys.argv) > 1 else None
    PosePipeline(exercise_id).run()