from abc import ABC, abstractmethod
from typing import Iterator

class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> str:
        pass

class BaseStreamModel(BaseModel):
    """ Base stream model class """
    @abstractmethod
    def stream(self, message: list[dict[str, str]]) -> Iterator[str]:
        """ Stream the response of the model
        - args:
            - message (`list[dict[str, str]]`): the message to be sent to the model
        - returns:
            - `Iterator[str]`: the response of the model
        """
        pass
