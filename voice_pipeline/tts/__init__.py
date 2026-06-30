from abc import ABC, abstractmethod


class TTSEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        ...
