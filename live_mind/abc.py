from abc import ABC, abstractmethod
from collections.abc import Generator

class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> str:
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
