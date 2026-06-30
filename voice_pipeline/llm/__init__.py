from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class LLMEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def generate(self, prompt: str) -> Iterator[str]:
        ...

    @abstractmethod
    def generate_sync(self, prompt: str) -> str:
        ...
