import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_REPO = "LiquidAI/LFM2.5-350M-GGUF"
_LFM_GGUF = "LFM2.5-350M-Q4_0.gguf"


def download_llm_model(
    dest_dir: Path,
    on_progress: Callable[[str, float], None] | None = None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _LFM_GGUF

    if dest_path.exists():
        logger.info(f"LLM model already cached: {dest_path}")
        return dest_path

    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading {_LFM_GGUF} from HuggingFace...")
    on_progress and on_progress(_LFM_GGUF, 0.0)

    def callback(current, total):
        on_progress and on_progress(_LFM_GGUF, current / total if total else 0.0)

    path = hf_hub_download(
        repo_id=_REPO,
        filename=_LFM_GGUF,
        local_dir=str(dest_dir),
        progress_callback=callback,
    )
    on_progress and on_progress(_LFM_GGUF, 1.0)
    return Path(path)
