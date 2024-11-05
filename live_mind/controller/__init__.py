""" Controllers for the LiveMind framework
usage:
pseudo code:

from live_mind import LMController

controller = LMController(...)

input_stream = ... # the input stream for the controller
stream_end = False

while not stream_end:
    input_stream.wait() # wait till the input_stream changes
    stream_end = input_stream.is_end()
    for response in controller(input_stream.text, stream_end=stream_end):
        print(response)

controller.reset()
"""

__all__ = [
    'LMController',
    'CompleteController',
    'LMStreamController',
    'CompleteStreamController',
]

from collections.abc import Callable, Generator
from . import abc
from ..abc import BaseModel, BaseStreamModel
from ..action import Action
from ..action.cache import SegmentActionCache, CacheEntry
from ..action.actions import Wait, Inference, Response
from ..formatter import BaseFormatter

class LMController(abc.BaseController):
    """ the LMController::
    args:
    - `segmenter`: `Callable[[str], list[str]]`: a function that segments the user input into prompts
    - `formatter`: `BaseFormatter`: the formatter for the prompts
    - `infer_model`: `BaseModel`: the model for the inference stage
    - `output_model`: `BaseModel`: the model for the output stage
    - `answer_format`: `str|None`: the format for the answer. The `answer_format` is append to the final prompt at the output stage. default: `None`
    - `logger`: `logging.Logger|None`: the logger for the controller, default: `None`
    """

    def __init__(
        self,
        segmenter: Callable[[str], list[str]],
        formatter: BaseFormatter,
        infer_model: BaseModel,
        output_model: BaseModel,
        hypothesize: bool = False,
        summarize_len: int = -1,
        answer_format: str|None = None,
    ):
        self.action_cache = SegmentActionCache()
        self.segmenter = segmenter
        self.formatter = formatter
        self.infer_model = infer_model
        self.output_model = output_model
        self.answer_format = answer_format
        self.action_types = [Wait, Inference]


    def  __call__(self, prompt: str, stream_end:bool=False) -> Generator[str, None, None]:
        """ Return the generator of responses from the LLM given the prompt """
        prompts = self.segmenter(prompt)
        if not stream_end:
            prompts = prompts[:-1]
        cache_entries, new_prompts = self.action_cache.read_action(prompts)
        if not new_prompts:
            return
        yield from self._step(cache_entries, new_prompts, stream_end)


    def _step(self, cache_entries: list[CacheEntry], new_prompts:list[str], stream_end: bool=False) -> Generator[str, None, None]:
        """ The main step for the controller, return the generator for the response of the LLM for the step.
        If `stream_end` is `True`, execute the output stage.
        Otherwise, execute the inference stage, and execute the hypothesis stage and the summarization stage if needed """
        if stream_end:
            response_action, response = self._output(cache_entries, new_prompts)
            self.action_cache.write_action([response_action,])
            yield response
        else:
            actions = []
            infer_action, response = self._inference(cache_entries, new_prompts)
            actions.append(infer_action)
            yield response
            self.action_cache.write_action(actions)


    def _inference(self, cache_entries, new_prompts: list[str]) -> tuple[Action, str]:
        """ execute the inference stage """
        msg = self.formatter.format_inference(cache_entries, new_prompts)
        response = self.infer_model.chat_complete(msg)
        # if the action is not parsed, write a wait as a placeholder to avoid frequent inference
        action = self.formatter.parse_action(response, self.action_types)
        if action is None:
            action = Action(type=Inference, content=response)
        return action, response


    def _output(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> tuple[Action, str]:
        """ execute the output stage """
        msg = self.formatter.format_output(cache_entries, new_prompts)
        if self.answer_format:
            if msg[-1]['role'] == 'user':
                msg[-1]['content'] += "\n\n"+self.answer_format
            elif msg[-2]['role'] == 'user':
                msg[-2]['content'] += "\n\n"+self.answer_format
            else:
                raise ValueError("The last two messages are not from the user.")
        response = self.output_model.chat_complete(msg)
        action = Action(type=Response, content=response)
        return action, response


    def reset(self):
        """ Reset the controller for a new conversation """
        self.action_cache.clear_cache()


class CompleteController(abc.BaseController):
    """ The CompleteCoTController is a controller that simulate conventional conversation with the LLM with complete prompts. """
    def __init__(
        self,
        formatter: BaseFormatter,
        output_model: BaseModel,
        answer_format: str|None = None
    ) -> None:
        self.formatter = formatter
        self.output_model = output_model
        self.answer_format = answer_format

    def __call__(self, prompt:str, stream_end:bool=False) -> Generator[str, None, None]:
        """ Return the generator for the response of the LLM given the prompt """
        if not stream_end:
            return
        msg = self.formatter.format_output([], [prompt,])
        if self.answer_format:
            msg[-1]['content'] += "\n\n"+self.answer_format
        yield self.output_model.chat_complete(msg)

    def reset(self):
        pass

"""
The StreamController is a controller that returns the response strings in a streaming fashion (TextStreamer)
They are used in the playground to display the responses in real-time.
"""
class LMStreamController(LMController, abc.BaseStreamController):
    """ The LMStreamController is a controller that returns the response strings in a streaming fashion (TextStreamer)
    This is useful for real-time display of the responses.
    
    The LMStreamController can be used as a LMController by calling the `__call__` method.
    For streaming, use the `iter_call` method.

    Compared to the LMController, the LMStreamController requires the models to be stream models (BaseStreamModel) with the `stream` method:
    - stream(self, message: list[dict[str, str]]) -> Generator[str, None, None]

    The iter_call method returns a generator of TextStreamer, for each yielded TextStreamer, the user should iterate over the TextStreamer
    and to exhauste it to get the response strings. If the TextStreamer is not exhausted, an Exception will be raised. (See the TextStreamer
    class in live_mind/controller/abc.py for more information)
    """
    def __init__(
        self,
        segmenter: Callable[[str], list[str]],
        formatter: BaseFormatter,
        infer_model: BaseStreamModel,
        output_model: BaseStreamModel,
        answer_format: str|None = None,
    ):
        self.action_cache = SegmentActionCache()
        self.segmenter = segmenter
        self.formatter = formatter
        self.infer_model: BaseStreamModel = infer_model
        self.output_model: BaseStreamModel = output_model
        self.answer_format = answer_format
        self.action_types = [Wait, Inference]


    def iter_call(
        self,
        prompt: str,
        stream_end:bool=False
    ) -> Generator[abc.RespnseStreamer, None, None]:
        prompts = self.segmenter(prompt)
        if not stream_end:
            prompts = prompts[:-1]
        cache_entries, new_prompts = self.action_cache.read_action(prompts)
        if not new_prompts:
            return
        yield from self._iter_step(cache_entries, new_prompts, stream_end)


    def _iter_step(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str],
        stream_end: bool=False
    ) -> Generator[abc.RespnseStreamer, None, None]:
        if stream_end:
            response_action = yield from self._iter_output(cache_entries, new_prompts)
            self.action_cache.write_action([response_action,])
        else:
            actions = []
            infer_action = yield from self._iter_inference(cache_entries, new_prompts)
            actions.append(infer_action)
            self.action_cache.write_action(actions)


    def _iter_inference(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> Generator[abc.RespnseStreamer, None, Action]:
        msg = self.formatter.format_inference(cache_entries, new_prompts)
        response_gen = self.infer_model.stream(msg)
        text_streamer = abc.RespnseStreamer(response_gen)
        yield text_streamer
        response = text_streamer.text
        action = self.formatter.parse_action(response, self.action_types)
        if action is None:
            action = Action(type=Inference, content=response)
        return action


    def _iter_output(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> Generator[abc.RespnseStreamer, None, Action]:
        msg = self.formatter.format_output(cache_entries, new_prompts)
        if self.answer_format:
            msg[-1]['content'] += "\n\n"+self.answer_format
        response_gen = self.output_model.stream(msg)
        text_streamer = abc.RespnseStreamer(response_gen)
        yield text_streamer
        response = text_streamer.text
        action = Action(type=Response, content=response)
        return action


class CompleteStreamController(abc.BaseStreamController):
    """ The CompleteCoTController is a controller that simulate conventional conversation with the LLM with complete prompts. """
    def __init__(
        self,
        formatter: BaseFormatter,
        output_model: BaseStreamModel,
        answer_format: str|None = None
    ) -> None:
        self.formatter = formatter
        self.output_model: BaseStreamModel = output_model
        self.answer_format = answer_format

    def __call__(self, prompt:str, stream_end:bool=False) -> Generator[str, None, None]:
        """ Return the generator for the response of the LLM given the prompt """
        if not stream_end:
            return
        msg = self.formatter.format_output([], [prompt,])
        if self.answer_format:
            msg[-1]['content'] += "\n\n"+self.answer_format
        yield self.output_model.chat_complete(msg)

    def iter_call(self, prompt: str, stream_end:bool=False) -> Generator[abc.RespnseStreamer, None, None]:
        if not stream_end:
            return
        msg = self.formatter.format_output([], [prompt,])
        if self.answer_format:
            msg[-1]['content'] += "\n\n"+self.answer_format
        response_gen = self.output_model.stream(msg)
        text_streamer = abc.RespnseStreamer(response_gen)
        yield text_streamer

    def reset(self):
        pass
