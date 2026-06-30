import logging
import os
from pathlib import Path
from typing import Iterator

from . import LLMEngine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are LFM, a helpful and concise voice assistant. "
    "Keep responses under 2 sentences. Be direct and natural."
)

_STOP_TOKENS = ["</|user|>", "</|system|>", "<|user|>", "<|system|>"]


class LFMEngine(LLMEngine):
    def __init__(
        self,
        model_path: Path | str | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_tokens: int = 128,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4
        self.max_tokens = max_tokens
        self._llm = None

    def load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            logger.warning(
                f"GGUF not found at {self.model_path}. "
                "Use ModelDownloader or set llm_model_path in PipelineConfig."
            )
            return

        from llama_cpp import Llama

        logger.info(f"Loading GGUF: {self.model_path}")
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        logger.info("LFM model loaded")

    def generate(self, prompt: str) -> Iterator[str]:
        if self._llm is None:
            yield "[LFM not loaded]"
            return

        formatted = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

        for chunk in self._llm(
            formatted,
            max_tokens=self.max_tokens,
            temperature=0.7,
            stop=["<|user|>", "<|system|>"],
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token and token not in _STOP_TOKENS:
                yield token

    def generate_sync(self, prompt: str) -> str:
        return "".join(self.generate(prompt))
