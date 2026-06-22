import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from detection.angle_calculator import AngleCalculator
from exercise.exercise_model import ExerciseModel
from exercise.exercise_state_machine import ExerciseStateMachine
from exercise.posture_rules import PostureRules
from user.user_profile import UserProfile

model = ExerciseModel()
print(f"Exercises: {model.list_exercises()}")

squat = model.get_exercise('squat')
sm = ExerciseStateMachine(squat)

profile = UserProfile(name="Test", fitness_level="beginner", limitations=["right_knee"], preferred_exercises=["squat"])
rules = PostureRules(squat, profile)

print(f"Profile modifier: {profile.threshold_modifier}")
print(f"Limited joints: {profile.limited_joints}")

angles = {'right_knee': 170, 'left_knee': 168, 'spine': 10, 'legs_spread': 30}
result = sm.update(angles)
print(f"Phase: {result['phase']}, Started: {result['started']}")

print("\nAll imports and paths working correctly!")
