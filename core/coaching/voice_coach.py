import queue
import threading
import time
import logging

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

from core.coaching.coach_messages import (
    get_start_message, get_phase_message, get_rep_message,
    get_error_message, get_session_end_message,
)

_ERROR_COOLDOWN        = 8.0   # seconds between same joint error
_PHASE_COOLDOWN        = 1.5
_REP_COOLDOWN          = 1.0


class VoiceCoach:
    def __init__(self, style: str = 'motivator', speech_rate: int = 175, enabled: bool = True):
        self.style   = style
        self.enabled = enabled and _TTS_AVAILABLE
        self._queue: queue.Queue = queue.Queue()
        self._cooldowns: dict    = {}
        self._running            = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="VoiceCoach")
        self._thread.start()
        self._speech_rate = speech_rate

        if not _TTS_AVAILABLE:
            logging.warning("pyttsx3 not installed — voice feedback disabled.")

    # ── public event hooks ────────────────────────────────────────────────

    def on_welcome(self, name: str, exercise_name: str):
        text = f"Hey {name}! Let's do some {exercise_name}. Get into starting position."
        self._say(text, key="welcome", cooldown=30.0)

    def on_start(self):
        self._say(get_start_message(self.style), key="start", cooldown=10.0)

    def on_phase_transition(self, phase_name: str, instruction: str):
        text = get_phase_message(self.style, phase_name, instruction)
        self._say(text, key=f"phase_{phase_name}", cooldown=_PHASE_COOLDOWN)

    def on_rep_complete(self, score: float):
        text = get_rep_message(self.style, score)
        self._say(text, key="rep", cooldown=_REP_COOLDOWN)

    def on_error(self, joint: str):
        text = get_error_message(self.style, joint)
        self._say(text, key=f"err_{joint}", cooldown=_ERROR_COOLDOWN)

    def on_session_end(self):
        text = get_session_end_message(self.style)
        self._say(text, key="end", cooldown=5.0)

    def shutdown(self):
        self._running = False
        self._thread.join(timeout=3.0)

    # ── internals ─────────────────────────────────────────────────────────

    def _say(self, text: str, key: str = None, cooldown: float = 0.0):
        if not self.enabled:
            logging.info(f"[Coach] {text}")
            return
        if key:
            now = time.time()
            if now - self._cooldowns.get(key, 0) < cooldown:
                return
            self._cooldowns[key] = now
        if self._queue.qsize() < 2:          # don't pile up
            self._queue.put(text)

    def _worker(self):
        engine = None
        if self.enabled:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', self._speech_rate)
                # prefer first English voice if available
                voices = engine.getProperty('voices')
                for v in voices:
                    if 'en' in (v.languages[0] if v.languages else '').lower() or 'en' in v.id.lower():
                        engine.setProperty('voice', v.id)
                        break
            except Exception as e:
                logging.warning(f"TTS init failed: {e}")
                engine = None

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if engine:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        logging.warning(f"TTS error: {e}")
                else:
                    logging.info(f"[Coach] {text}")
                self._queue.task_done()
            except queue.Empty:
                continue
