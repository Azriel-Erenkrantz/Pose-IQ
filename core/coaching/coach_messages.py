import random

_MESSAGES = {
    'tough': {
        'start': [
            "Move it!", "Let's go — no excuses!", "Get moving, now!"
        ],
        'phase': {
            'descending': ["Down!", "Lower yourself!", "Go down, now!"],
            'hold':       ["Hold it!", "Stay there!", "Don't move!"],
            'ascending':  ["Up!", "Push it up!", "Rise!"],
            'standing':   ["Reset!", "Stand straight!"],
            'curling':    ["Curl it!", "Pull up!"],
            'lowering':   ["Control it down!", "Slow down!"],
            'pressing':   ["Press it up!", "Push overhead!"],
            'setup':      ["Get in position!"],
            'start':      ["Set position!"],
        },
        'rep': {
            'good': ["Good. Keep going.", "Next one.", "Don't stop."],
            'bad':  ["Sloppy. Fix your form.", "Again — with more focus.", "That form was weak."],
        },
        'error': {
            'spine':         ["Back straight, now!", "Don't slouch!", "Spine up!"],
            'right_knee':    ["Fix that right knee!", "Right knee — align it!"],
            'left_knee':     ["Fix that left knee!", "Left knee — align it!"],
            'right_elbow':   ["Right elbow position!", "Fix that right elbow!"],
            'left_elbow':    ["Left elbow position!", "Fix that left elbow!"],
            'right_arm_body':["Keep that right arm in!", "Right arm flaring out!"],
            'left_arm_body': ["Keep that left arm in!", "Left arm flaring out!"],
            'legs_spread':   ["Watch your foot width!", "Adjust your stance!"],
            'generic':       ["Fix your form!", "That's wrong — correct it!"],
        },
        'encouragement': [
            "Push through it.", "You're not done yet.", "Dig deeper.", "More!"
        ],
        'session_end': [
            "Done. Same time tomorrow.", "Not bad. Keep training.", "Session over."
        ],
    },

    'calm': {
        'start': [
            "Begin whenever you're ready.", "Take your time and start.", "Let's begin, nice and steady."
        ],
        'phase': {
            'descending': ["Gently lower down.", "Ease into it.", "Slowly go down."],
            'hold':       ["Hold this position.", "Stay here a moment.", "Good, hold it."],
            'ascending':  ["Come back up.", "Rise smoothly.", "Return to standing."],
            'standing':   ["Back to standing.", "Reset your position."],
            'curling':    ["Curl up slowly.", "Bring it up."],
            'lowering':   ["Lower with control.", "Ease it back down."],
            'pressing':   ["Press overhead.", "Extend your arms."],
            'setup':      ["Get into position."],
            'start':      ["Hold the starting position."],
        },
        'rep': {
            'good': ["Well done.", "Nice rep.", "Good work, continue."],
            'bad':  ["Let's focus on form.", "Take care with your alignment.", "Adjust and try again."],
        },
        'error': {
            'spine':         ["Keep your back straight.", "Maintain a neutral spine.", "Straighten your back."],
            'right_knee':    ["Watch your right knee.", "Align your right knee."],
            'left_knee':     ["Watch your left knee.", "Align your left knee."],
            'right_elbow':   ["Adjust your right elbow.", "Check your right elbow angle."],
            'left_elbow':    ["Adjust your left elbow.", "Check your left elbow angle."],
            'right_arm_body':["Keep your right arm in.", "Right arm staying close to body."],
            'left_arm_body': ["Keep your left arm in.", "Left arm staying close to body."],
            'legs_spread':   ["Check your foot position.", "Adjust your stance width."],
            'generic':       ["Check your form.", "Adjust your position."],
        },
        'encouragement': [
            "You're doing well.", "Keep it up.", "Steady progress.", "Good effort."
        ],
        'session_end': [
            "Nice session. Rest well.", "Good work today.", "Session complete. Take care."
        ],
    },

    'motivator': {
        'start': [
            "You've got this!", "Let's crush it!", "Here we go — give it everything!"
        ],
        'phase': {
            'descending': ["Down you go!", "Drop into it!", "Sink in!"],
            'hold':       ["Hold strong!", "You're doing great, stay!", "Power through the hold!"],
            'ascending':  ["Drive up!", "Push the floor away!", "Come on, rise up!"],
            'standing':   ["Reset and go again!", "Stand tall, warrior!"],
            'curling':    ["Curl it up!", "Show those biceps!"],
            'lowering':   ["Control it down, feel the burn!", "Slow and steady on the way down!"],
            'pressing':   ["Press it to the sky!", "Send it overhead!"],
            'setup':      ["Get ready to crush it!"],
            'start':      ["Lock in that position!"],
        },
        'rep': {
            'good': ["Amazing!", "That's what I'm talking about!", "Crushing it!"],
            'bad':  ["You can do better — let's go!", "Focus up, you've got this!", "Next one — believe in yourself!"],
        },
        'error': {
            'spine':         ["Back straight — you're almost there!", "Core tight, back straight!"],
            'right_knee':    ["Mind that right knee — keep it aligned!"],
            'left_knee':     ["Mind that left knee — keep it aligned!"],
            'right_elbow':   ["Right elbow — nail that angle!"],
            'left_elbow':    ["Left elbow — nail that angle!"],
            'right_arm_body':["Right arm tucked in — you've got this!"],
            'left_arm_body': ["Left arm tucked in — keep it tight!"],
            'legs_spread':   ["Adjust that stance — you're so close!"],
            'generic':       ["Fix that form — you're stronger than this!"],
        },
        'encouragement': [
            "Keep pushing!", "You're on fire!", "Nothing can stop you!", "Beast mode activated!"
        ],
        'session_end': [
            "Incredible session! You're getting stronger!", "That's how it's done!", "Beast. See you next time!"
        ],
    },
}


def get_start_message(style: str) -> str:
    return random.choice(_MESSAGES.get(style, _MESSAGES['motivator'])['start'])


def get_phase_message(style: str, phase_name: str, instruction: str) -> str:
    msgs = _MESSAGES.get(style, _MESSAGES['motivator'])['phase']
    if phase_name in msgs:
        return random.choice(msgs[phase_name])
    return instruction


def get_rep_message(style: str, score: float) -> str:
    category = 'good' if score >= 75 else 'bad'
    return random.choice(_MESSAGES.get(style, _MESSAGES['motivator'])['rep'][category])


def get_error_message(style: str, joint: str) -> str:
    error_msgs = _MESSAGES.get(style, _MESSAGES['motivator'])['error']
    options = error_msgs.get(joint) or error_msgs['generic']
    return random.choice(options)


def get_session_end_message(style: str) -> str:
    return random.choice(_MESSAGES.get(style, _MESSAGES['motivator'])['session_end'])
