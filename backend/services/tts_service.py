"""Text-to-Speech service — generates audio for predicted signs.

Uses pyttsx3 (offline) when available, falls back to gTTS (online),
and ultimately the browser Web Speech API is the primary TTS mechanism.
"""

import os
import threading
import time
from pathlib import Path

_TTS_ENGINE = None  # 'pyttsx3', 'gtts', or None

try:
    import pyttsx3
    _TTS_ENGINE = "pyttsx3"
except Exception:
    try:
        from gtts import gTTS
        _TTS_ENGINE = "gtts"
    except Exception:
        pass

AUDIO_DIR = Path(__file__).parent.parent / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class TTSService:
    """Thread-safe text-to-speech service."""

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = True
        self._rate = 150
        self._volume = 1.0
        self._engine_name = _TTS_ENGINE
        self._latest_file: str | None = None
        # pyttsx3 engine is created per-thread inside _generate() to avoid Windows
        # COM STA threading violation (engine created on main thread, runAndWait()
        # called from daemon thread → COM marshal deadlock on asyncio ProactorEventLoop)

    @property
    def available(self) -> bool:
        return self._engine_name is not None

    @property
    def engine_name(self) -> str:
        return self._engine_name or "none"

    def speak(self, text: str) -> str | None:
        """Generate audio file for the given text. Returns relative URL or None."""
        if not self._enabled or not self._engine_name or not text:
            return None

        # Run generation in background thread so it doesn't block the event loop
        thread = threading.Thread(target=self._generate, args=(text,), daemon=True)
        thread.start()

        # Return the expected URL path (file may still be generating)
        return "/static/audio/latest.wav" if self._engine_name == "pyttsx3" else "/static/audio/latest.mp3"

    def _generate(self, text: str):
        with self._lock:
            try:
                if self._engine_name == "pyttsx3":
                    import pyttsx3 as _pyttsx3
                    # Create the engine on THIS thread so COM STA object is thread-local
                    engine = _pyttsx3.init()
                    engine.setProperty("rate", self._rate)
                    engine.setProperty("volume", self._volume)
                    out = str(AUDIO_DIR / "latest.wav")
                    engine.save_to_file(text, out)
                    engine.runAndWait()
                    engine.stop()
                    self._latest_file = out
                elif self._engine_name == "gtts":
                    from gtts import gTTS
                    out = str(AUDIO_DIR / "latest.mp3")
                    tts = gTTS(text=text, lang="en", slow=False)
                    tts.save(out)
                    self._latest_file = out
            except Exception as e:
                print(f"TTS generation error: {e}")

    def get_latest_path(self) -> str | None:
        """Return absolute path to the latest generated audio file, if it exists."""
        if self._latest_file and os.path.isfile(self._latest_file):
            return self._latest_file
        # Check for any existing file
        for ext in ("wav", "mp3"):
            p = AUDIO_DIR / f"latest.{ext}"
            if p.is_file():
                return str(p)
        return None

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def set_rate(self, rate: int):
        self._rate = rate  # applied on next _generate() call

    def set_volume(self, volume: float):
        self._volume = max(0.0, min(1.0, volume))  # applied on next _generate() call

    def get_settings(self) -> dict:
        return {
            "tts_enabled": self._enabled,
            "tts_engine": self.engine_name,
            "tts_rate": self._rate,
            "tts_volume": self._volume,
        }
