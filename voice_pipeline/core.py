import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import PipelineConfig, PipelineResult
from .downloader import download_llm_model

logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._asr = None
        self._llm = None
        self._tts = None
        self._loaded = False

        self.on_transcript: Callable[[str], None] | None = None
        self.on_token: Callable[[str], None] | None = None
        self.on_audio: Callable[[bytes], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self.on_download_progress: Callable[[str, float], None] | None = None

    def initialize(self) -> None:
        t0 = time.time()

        self._init_asr()
        self._init_llm()
        self._init_tts()

        self._loaded = True
        logger.info(f"Pipeline initialized in {time.time() - t0:.1f}s")

    def _init_asr(self) -> None:
        try:
            from .asr.whisper_asr import FasterWhisperASR

            engine = FasterWhisperASR()
            self._notify("Loading ASR...")
            engine.load()
            self._asr = engine
        except ImportError:
            logger.warning("faster-whisper not installed. ASR disabled.")
        except Exception as e:
            self._notify_error(e)
            logger.warning(f"ASR init failed: {e}")

    def _init_llm(self) -> None:
        try:
            from .llm.lfm_engine import LFMEngine

            model_path = self.config.llm_model_path
            if model_path is None:
                model_path = download_llm_model(
                    self.config.models_dir, self._on_download_progress
                )
                self.config.llm_model_path = model_path

            engine = LFMEngine(model_path=model_path)
            self._notify("Loading LLM...")
            engine.load()
            self._llm = engine
        except ImportError:
            logger.warning("llama-cpp-python not installed. LLM disabled.")
        except Exception as e:
            self._notify_error(e)
            logger.warning(f"LLM init failed: {e}")

    def _init_tts(self) -> None:
        try:
            from .tts.supertonic_tts import SupertonicTTS

            engine = SupertonicTTS(voice=self.config.tts_voice)
            self._notify("Loading TTS...")
            engine.load()
            self._tts = engine
        except ImportError:
            logger.warning("supertonic not installed. TTS disabled.")
        except Exception as e:
            self._notify_error(e)
            logger.warning(f"TTS init failed: {e}")

    def process_audio(self, audio_data: bytes, sample_rate: int = 16000) -> PipelineResult | None:
        if not self._loaded:
            logger.error("Pipeline not initialized. Call initialize() first.")
            return None

        raw = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        transcript = ""
        if self._asr:
            try:
                self._notify("Transcribing...")
                transcript = self._asr.transcribe(raw, sample_rate)
                self._notify_transcript(transcript)
            except Exception as e:
                self._notify_error(e)
                logger.error(f"ASR error: {e}")

        response = ""
        if self._llm and transcript:
            try:
                self._notify("Generating...")
                for token in self._llm.generate(transcript):
                    response += token
                    self._notify_token(token)
            except Exception as e:
                self._notify_error(e)
                logger.error(f"LLM error: {e}")

        audio_bytes: bytes | None = None
        if self._tts and response:
            try:
                self._notify("Synthesizing...")
                audio_bytes = self._tts.synthesize(response)
                if audio_bytes:
                    self._notify_audio(audio_bytes)
            except Exception as e:
                self._notify_error(e)
                logger.error(f"TTS error: {e}")

        return PipelineResult(
            transcript=transcript,
            response=response,
            audio=audio_bytes,
        )

    def release(self) -> None:
        self._asr = None
        self._llm = None
        self._tts = None
        self._loaded = False
        logger.info("Pipeline released")

    def _on_download_progress(self, name: str, progress: float) -> None:
        if self.on_download_progress:
            self.on_download_progress(name, progress)

    def _notify(self, msg: str) -> None:
        logger.info(msg)

    def _notify_transcript(self, text: str) -> None:
        if self.on_transcript:
            self.on_transcript(text)

    def _notify_token(self, token: str) -> None:
        if self.on_token:
            self.on_token(token)

    def _notify_audio(self, audio: bytes) -> None:
        if self.on_audio:
            self.on_audio(audio)

    def _notify_error(self, error: Exception) -> None:
        if self.on_error:
            self.on_error(error)
