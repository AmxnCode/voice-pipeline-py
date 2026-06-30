from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    llm_model_path: Path | None = None
    tts_voice: str = "F5"
    max_tokens: int = 1024
    temperature: float = 0.7
    models_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "voice-pipeline")


@dataclass
class PipelineResult:
    transcript: str
    response: str
    audio: bytes | None = None
