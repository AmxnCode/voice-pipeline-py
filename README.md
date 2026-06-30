# Voice Pipeline — Python SDK

A fully local voice pipeline: **ASR + LLM + TTS** on-device, in Python.

| Component | Engine | Runtime | Package |
|-----------|--------|---------|---------|
| **ASR** | Whisper small | CTranslate2 | `faster-whisper` |
| **LLM** | LFM 2.5-350M (Q4_0) | llama.cpp | `llama-cpp-python` |
| **TTS** | Supertonic 3 | ONNX | `supertonic` |

Models auto-download on first run (~1.8 GB total).

## Install

```bash
pip install "voice-pipeline[all] @ git+https://github.com/AmxnCode/voice-pipeline-py.git"
```

Or install components individually:

```bash
pip install "voice-pipeline[asr]"    # ASR only
pip install "voice-pipeline[llm]"    # LLM only
pip install "voice-pipeline[tts]"    # TTS only
```

## Quick Start

```python
from voice_pipeline import VoicePipeline, PipelineConfig

pipeline = VoicePipeline()

pipeline.on_transcript = lambda t: print(f"User: {t}")
pipeline.on_token = lambda t: print(t, end="", flush=True)
pipeline.on_audio = lambda w: open("output.wav", "wb").write(w)
pipeline.on_error = lambda e: print(f"Error: {e}")

pipeline.initialize()

result = pipeline.process_audio(audio_data)
pipeline.release()
```

## API

### VoicePipeline

```python
class VoicePipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig())

    def initialize(self) -> None
    def process_audio(audio_data: bytes, sample_rate: int = 16000) -> PipelineResult | None
    def release() -> None

    on_transcript: Callable[[str], None] | None
    on_token: Callable[[str], None] | None
    on_audio: Callable[[bytes], None] | None
    on_error: Callable[[Exception], None] | None
    on_download_progress: Callable[[str, float], None] | None
```

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    llm_model_path: Path | None = None     # Override GGUF path
    tts_voice: str = "M1"                  # Voice style
    max_tokens: int = 512
    temperature: float = 0.7
    models_dir: Path = ~/.cache/voice-pipeline/
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    transcript: str
    response: str
    audio: bytes | None
```

## Requirements

- Python 3.10+
- ~2 GB free storage

## Architecture

```
Audio In → faster-whisper → LFM 350M (llama.cpp) → Supertonic 3 (ONNX) → Audio Out
           └── ASR ───────┘ └── Text Gen ──────────┘ └── TTS ────────────┘
```

## License

Apache 2.0
