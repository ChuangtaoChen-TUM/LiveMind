from abc import ABC, abstractmethod
from collections.abc import Generator

class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> str:
        pass

class BaseStreamModel(BaseModel):
    """ Base stream model class """
    @abstractmethod
    def stream(self, message: list[dict[str, str]]) -> Generator[str, None, None]:
        """ Stream the response of the model
        - args:
            - message (`list[dict[str, str]]`): the message to be sent to the model
        - returns:
            - `Generator[str, None, None]`: the generator for the response of the model
        """
        pass


class BaseController(ABC):
    """ The base class for the controller in the live_mind framework:
     the controller is responsible for generating the prompts to the LLM based on the user input.
     if no prompt is generated, the controller should return `None`.

    methods:
    - `__call__`: generate the prompts based on the user input, use `stream_end` to indicate the end of the stream, return `None` if no prompt is generated.
    """
    @abstractmethod
    def __call__(self, prompt:str, stream_end:bool=False) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def reset(self):
        pass


class BaseStreamController(BaseController):
    """ The base class for the stream controller in the live_mind framework:
     the stream controller is responsible for generating the prompts to the LLM based on the user input in a streaming fashion.
     if no prompt is generated, the controller should return `None`.

    methods:
    - `__call__`: generate the prompts based on the user input, use `stream_end` to indicate the end of the stream, return `None` if no prompt is generated.
    """
    @abstractmethod
    def iter_call(self, prompt:str, stream_end:bool=False) -> Generator['TextStreamer', None, None]:
        pass


class TextStreamer:
    def __init__(self, text_generator: Generator[str, None, None]):
        self.text_generator = text_generator
        self._text = ""
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self._exhausted:
            raise StopIteration
        try:
            next_text = next(self.text_generator)
            self._text += next_text
            return next_text
        except StopIteration:
            self._exhausted = True
            raise

    @property
    def text(self):
        if self._exhausted:
            return self._text
        else:
            raise ValueError("The generator is not exhausted yet")