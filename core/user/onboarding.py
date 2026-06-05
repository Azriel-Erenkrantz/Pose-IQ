from user.user_profile import UserProfile, FITNESS_LEVELS, LIMITATION_OPTIONS
from exercise.exercise_model import ExerciseModel
from coaching.coach_messages import COACH_STYLES


def run_onboarding() -> UserProfile:
    print("\n" + "=" * 50)
    print("  Welcome to Pose-IQ!")
    print("  Let's set up your profile")
    print("=" * 50)

    name = input("\nWhat's your name? ").strip()
    if not name:
        name = "User"

    print("\nFitness level:")
    levels = list(FITNESS_LEVELS.keys())
    for i, level in enumerate(levels, 1):
        info = FITNESS_LEVELS[level]
        print(f"  {i}. {info['label']}")
    while True:
        choice = input("Choose (1-3): ").strip()
        if choice in ['1', '2', '3']:
            fitness_level = levels[int(choice) - 1]
            break
        print("Please enter 1, 2, or 3")

    print("\nDo you have any physical limitations?")
    print("  0. None")
    for i, limitation in enumerate(LIMITATION_OPTIONS, 1):
        print(f"  {i}. {limitation.replace('_', ' ').title()}")
    print("Enter numbers separated by commas (e.g., 1,3), or 0 for none:")
    limitations = []
    choice = input("Choose: ").strip()
    if choice != '0' and choice:
        for num in choice.split(','):
            num = num.strip()
            if num.isdigit() and 1 <= int(num) <= len(LIMITATION_OPTIONS):
                limitations.append(LIMITATION_OPTIONS[int(num) - 1])

    model = ExerciseModel()
    available = model.list_exercises()
    print("\nWhich exercises do you prefer?")
    print("  0. All exercises")
    for i, ex_id in enumerate(available, 1):
        ex = model.get_exercise(ex_id)
        print(f"  {i}. {ex.name}")
    print("Enter numbers separated by commas (e.g., 1,3,5), or 0 for all:")
    preferred = []
    choice = input("Choose: ").strip()
    if choice == '0' or not choice:
        preferred = available
    else:
        for num in choice.split(','):
            num = num.strip()
            if num.isdigit() and 1 <= int(num) <= len(available):
                preferred.append(available[int(num) - 1])
    if not preferred:
        preferred = available

    print("\nCoach style:")
    style_info = {
        'tough':     'Direct and demanding — short commands, no fluff',
        'calm':      'Gentle and patient — smooth, reassuring cues',
        'motivator': 'Energetic and encouraging — hype you up',
    }
    for i, s in enumerate(COACH_STYLES, 1):
        print(f"  {i}. {s.capitalize():12} — {style_info[s]}")
    while True:
        choice = input("Choose (1-3): ").strip()
        if choice in ['1', '2', '3']:
            coach_style = COACH_STYLES[int(choice) - 1]
            break
        print("Please enter 1, 2, or 3")

    profile = UserProfile(
        name=name,
        fitness_level=fitness_level,
        limitations=limitations,
        preferred_exercises=preferred,
        coach_style=coach_style,
    )
    profile.save()

    print(f"\nProfile saved for {name}!")
    print(f"  Level:      {FITNESS_LEVELS[fitness_level]['label']}")
    print(f"  Coach:      {coach_style.capitalize()}")
    print(f"  Limitations:{' ' + ', '.join(limitations) if limitations else ' None'}")
    print(f"  Exercises:  {', '.join(preferred)}")
    print()

    return profile


if __name__ == "__main__":
    run_onboarding()
