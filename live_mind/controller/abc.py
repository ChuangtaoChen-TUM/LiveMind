__all__ = [
    'BaseController',
    'BaseStreamController',
    'RespnseStreamer'
]

from abc import ABC, abstractmethod
from collections.abc import Iterator, Generator

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
    it can be used as a base controller
    it also supports the `iter_call` method, which is used the same way as the `__call__` method
    the main different is that instead returning a generator of complete response strings, the `iter_call` method returns
    a generator of `TextStreamer` objects, which can be used to stream the response strings.

    The TextStreamer can be used as an iterator to get the response string pieces.

    methods:
    - `__call__`: generate the prompts based on the user input, use `stream_end` to indicate the end of the stream, return `None` if no prompt is generated.
    """
    @abstractmethod
    def iter_call(self, prompt:str, stream_end:bool=False) -> Generator['RespnseStreamer', None, None]:
        pass


class RespnseStreamer:
    """ A wrapper for the generator of response strings to record the response text """
    def __init__(self, text_generator: Iterator[str]):
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
        if not self._exhausted:
            self._text += "".join(self.text_generator)
            self._exhausted = True
        return self._text

    @property
    def exhausted(self):
        return self._exhausted

