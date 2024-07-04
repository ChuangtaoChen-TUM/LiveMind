from collections.abc import Callable
from abc import ABC, abstractmethod
import logging
from .action.cache import SegmentActionCache
from .action.abc import BaseActionFormatter

class BaseController(ABC):
    """ The base class for the controller in the live_mind framework:
     the controller is responsible for generating the prompts to the LLM based on the user input.
     if no prompt is generated, the controller should return `None`.

    methods:
    - `__call__`: generate the prompts based on the user input, use `stream_end` to indicate the end of the stream, return `None` if no prompt is generated.
    - `update`: update the controller with the response from the LLM.
    - `reset`: reset the controller for a new conversation.
    """
    @abstractmethod
    def __call__(self, prompt:str, stream_end:bool=False) -> list[dict[str, str]]|None:
        pass

    @abstractmethod
    def update(self, response:str):
        pass

    @abstractmethod
    def reset(self):
        pass


class LMController(BaseController):
    """ the LMController maintains a cache of actions and generates prompts based on current prompt and cached actions.

    If the actions correspond to the prompt are in the cache, the controller returns `None`,
    otherwise, it will generate the prompts based on the actions and the prompt.
    """
    # TODO: support multi-round conversation
    def __init__(
        self,
        segmenter: Callable[[str], list[str]],
        formatter: BaseActionFormatter,
        drop_last: bool = True, # depends on the segmenter. If the last segment is incomplete, drop it.
        logger: logging.Logger|None = None
    ):
        self.action_cache = SegmentActionCache(segmenter, drop_last=drop_last)
        self.formatter = formatter
        self.drop_last = drop_last
        self.logger = logger

    def  __call__(self, prompt, stream_end:bool=False) -> list[dict[str, str]]|None:
        """ Generate the prompts based on the user input and the cached actions """
        if stream_end:
            if self.drop_last:
                self.action_cache.drop_last = False # do not drop the last segment
            actions, prompts = self.action_cache.read_action(prompt)
            if len(prompts) > 1 and self.logger:
                self.logger.info(f"Multiple prompts are generated: {prompts}")
            self.action_cache.drop_last = self.drop_last
            msg = self.formatter.format_output(actions, prompts)
            return msg

        actions, prompts = self.action_cache.read_action(prompt)
        if not prompts:
            return None
        if len(prompts) > 1 and self.logger:
            self.logger.info(f"Multiple prompts are generated: {prompts}")
        msg = self.formatter.format_inference(actions, prompts)
        return msg

    def update(self, response:str):
        """ Update the controller with the response from the LLM, if the response contains an action, write the action to the cache """
        action = self.formatter.parse_action(response)
        if action is not None:
            self.action_cache.write_action(action, prompt=None)
        elif self.logger:
            self.logger.warning(f"Action is not parsed from the response: {response}")

    def reset(self):
        """ Reset the controller for a new conversation """
        self.action_cache.clear_cache()


class CompleteCoTController(BaseController):
    """ The CompleteCoTController is a controller that simulate conventional conversation with the LLM with complete prompts. """
    def __init__(self, formatter: BaseActionFormatter) -> None:
        self.formatter = formatter

    def __call__(self, prompt:str, stream_end:bool=False):
        """ Only generate the prompts when the stream ends """
        if not stream_end:
            return None
        msg = self.formatter.format_output([], [prompt,])
        return msg

    def update(self, response:str):
        raise NotImplementedError
    
    def reset(self):
        pass
