import io
import threading

import pygame
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from app import config

# Per-emotion voice settings
# stability:  lower = more expressive/variable, higher = monotone
# style:      0–1 style exaggeration
# speed:      1.0 = normal, <1 slower, >1 faster
_EMOTION_SETTINGS: dict[str, VoiceSettings] = {
    "happy": VoiceSettings(
        stability=0.25,
        similarity_boost=0.75,
        style=0.70,
        speed=1.10,
        use_speaker_boost=True,
    ),
    "sad": VoiceSettings(
        stability=0.75,
        similarity_boost=0.75,
        style=0.10,
        speed=0.85,
        use_speaker_boost=False,
    ),
    "angry": VoiceSettings(
        stability=0.15,
        similarity_boost=0.75,
        style=0.90,
        speed=1.05,
        use_speaker_boost=True,
    ),
    "neutral": VoiceSettings(
        stability=0.50,
        similarity_boost=0.75,
        style=0.00,
        speed=1.00,
        use_speaker_boost=False,
    ),
}


class TTSClient:
    """Wraps ElevenLabs TTS. speak() is blocking — call from a worker thread.
    Pass emotion= to apply matching voice settings."""

    def __init__(self):
        self._client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY) if config.ELEVENLABS_API_KEY else None
        # Lock prevents overlapping audio if speak() is called twice in quick succession.
        self._lock   = threading.Lock()
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    def speak(self, text: str, emotion: str = "neutral") -> None:
        if not text.strip():
            return
        if self._client is None:
            print(f"TTS (no API key): {text!r}")
            return

        voice_settings = _EMOTION_SETTINGS.get(emotion, _EMOTION_SETTINGS["neutral"])

        with self._lock:
            try:
                audio_generator = self._client.text_to_speech.convert(
                    voice_id=config.ELEVENLABS_VOICE_ID,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    voice_settings=voice_settings,
                )
                audio_bytes = b"".join(audio_generator)
                audio_file  = io.BytesIO(audio_bytes)
                pygame.mixer.music.load(audio_file, "mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
            except Exception as exc:
                print(f"TTSClient error: {exc}")
