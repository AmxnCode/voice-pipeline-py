import io
import logging

import numpy as np
import soundfile as sf

from . import TTSEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


class SupertonicTTS(TTSEngine):
    def __init__(self, voice: str = "M1", model: str = "supertonic-3"):
        self.voice = voice
        self.model = model
        self._tts = None
        self._style = None
        self._sample_rate = SAMPLE_RATE

    def load(self) -> None:
        from supertonic import TTS

        logger.info(f"Loading Supertonic {self.model}...")
        self._tts = TTS(model=self.model, auto_download=True)
        self._style = self._tts.get_voice_style(voice_name=self.voice)
        self._sample_rate = self._tts.sample_rate
        logger.info(f"Supertonic ready (voice={self.voice})")

    def synthesize(self, text: str) -> bytes:
        if self._tts is None or not text.strip():
            return b""

        try:
            wav, _ = self._tts.synthesize(
                text=text,
                voice_style=self._style,
                lang="en",
                total_steps=8,
                speed=1.0,
            )

            if wav.ndim > 1:
                wav = wav.squeeze(0)

            buf = io.BytesIO()
            sf.write(buf, wav, self._sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Supertonic synthesis error: {e}")
            return b""
