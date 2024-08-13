__all__ = ['BaseDataset']

from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def select(
        self,
        num: int,
        randomize:bool,
        seed:int,
        split:str
    ) -> None:
        pass

    @abstractmethod
    def verify_answer(self, response: str, answer_text) -> bool:
        pass

    @abstractmethod
    def add_str(self, entry: dict) -> str|None:
        pass

    @property
    @abstractmethod
    def selected_questions(self) -> list:
        pass

    @property
    @abstractmethod
    def answer_format(self) -> str:
        pass
