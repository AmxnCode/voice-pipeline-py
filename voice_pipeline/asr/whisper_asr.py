import logging

import numpy as np

from . import ASREngine

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASREngine):
    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        logger.info(f"Loading faster-whisper {self.model_size}...")
        compute = "float16" if self.device not in ("cpu", "auto") else "int8"
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute,
        )
        logger.info("faster-whisper loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            return "[Whisper not loaded]"

        if sample_rate != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
        segments, _ = self._model.transcribe(audio)
        return " ".join(seg.text for seg in segments).strip()
