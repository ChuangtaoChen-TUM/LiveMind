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
    'CompleteCoTController',
    'action',
    'format',
    'text',
    'utils'
]

from collections.abc import Callable, Generator
import logging
from .abc import (
    BaseController,
    BaseStreamController,
    BaseModel,
    BaseStreamModel,
    TextStreamer
)
from ..action import Action
from ..action.cache import SegmentActionCache, CacheEntry
from ..action.actions import Wait, Inference, Hypothesize, Summarize, Response
from ..format import BaseFormatter
from .. import action
from .. import format
from .. import text
from .. import utils

class LMController(BaseController):
    """ the LMController::
    - args:
        - `segmenter`: `Callable[[str], list[str]]`: a function that segments the user input into prompts
        - `formatter`: `BaseFormatter`: the formatter for the prompts
        - `infer_model`: `BaseModel`: the model for the inference stage
        - `output_model`: `BaseModel`: the model for the output stage
        - `hypothesize`: `bool`: whether to hypothesize, default: `False`
        - `retreive_all`: `bool`: whether to retreive all the prompts, default: `True`. If `False`, only update one prompt segment at a time.
        - `summarize_len`: `int`: the number of prompts to summarize, default: `-1` (no summarization)
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
        retreive_all: bool = True,
        summarize_len: int = -1,
        answer_format: str|None = None,
        logger: logging.Logger|None = None
    ):
        self.action_cache = SegmentActionCache()
        self.segmenter = segmenter
        self.formatter = formatter
        self.infer_model = infer_model
        self.output_model = output_model
        self.hypothesize = hypothesize
        self.summarize_len = summarize_len
        self.retreive_all = retreive_all
        self.logger = logger
        self.answer_format = answer_format
        self.action_types = [Wait, Inference]
        if self.hypothesize:
            self.action_types.append(Hypothesize)
        if self.summarize_len > 0:
            self.action_types.append(Summarize)


    def  __call__(self, prompt: str, stream_end:bool=False) -> Generator[str, None, None]:
        """ Return the generator of responses from the LLM given the prompt """
        prompts = self.segmenter(prompt)
        if not stream_end:
            prompts = prompts[:-1]
        cache_entries, new_prompts = self.action_cache.read_action(prompts)
        if not new_prompts:
            return
        if self.retreive_all:
            yield from self._step(cache_entries, new_prompts, stream_end)
        else:
            start_index = len(prompts) - len(new_prompts)
            for i in range(len(new_prompts) - 1):
                cache_entries, new_prompts = self.action_cache.read_action(prompts[:start_index+i+1])
                yield from self._step(cache_entries, new_prompts, stream_end=False)
            cache_entries, new_prompts = self.action_cache.read_action(prompts)
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

            if self.hypothesize and infer_action.type == Wait:
                action, response = self._hypothesize(cache_entries, new_prompts)
                if action:
                    actions.append(action)
                yield response

            self.action_cache.write_action(actions)

            if self.summarize_len > 0:
                new_entry = CacheEntry(actions=actions, prompts=new_prompts)
                new_entries = cache_entries + [new_entry] # add generated actions in this step for summarization
                num_non_wait = len([
                    action for cache_entry in new_entries 
                    for action in cache_entry.actions if action.type != Wait]
                )
                if num_non_wait >= self.summarize_len:
                    sum_action, summarization = self._summarize(new_entries, [])
                    if sum_action:
                        actions.append(sum_action)
                    if summarization:
                        if sum_action:
                            self.action_cache.reset_action([sum_action, ])
                        yield summarization


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
            msg[-1]['content'] += "\n\n"+self.answer_format
        response = self.output_model.chat_complete(msg)
        action = Action(type=Response, content=response)
        return action, response


    def _hypothesize(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> tuple[Action, str]:
        """ Execute the hypothesis stage """
        msg = self.formatter.format_hypothesize(cache_entries, new_prompts)
        response = self.infer_model.chat_complete(msg)
        action = self.formatter.parse_action(response, self.action_types)
        if not action:
            action = Action(type=Inference, content=response)
        return action, response


    def _summarize(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> tuple[Action|None, str]:
        """ Execute the summarization stage """
        msg = self.formatter.format_summarize(cache_entries, new_prompts)
        response = self.infer_model.chat_complete(msg)
        action = self.formatter.parse_action(response, self.action_types)
        return action, response


    def reset(self):
        """ Reset the controller for a new conversation """
        self.action_cache.clear_cache()


class LMStreamController(LMController, BaseStreamController):
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
        hypothesize: bool = False,
        retreive_all: bool = True,
        summarize_len: int = -1,
        answer_format: str|None = None,
        logger: logging.Logger|None = None
    ):
        super().__init__(
            segmenter=segmenter,
            formatter=formatter,
            infer_model=infer_model,
            output_model=output_model,
            hypothesize=hypothesize,
            retreive_all=retreive_all,
            summarize_len=summarize_len,
            answer_format=answer_format,
            logger=logger
        )
    
    def iter_call(
        self,
        prompt: str,
        stream_end:bool=False
    ) -> Generator[TextStreamer, None, None]:
        prompts = self.segmenter(prompt)
        if not stream_end:
            prompts = prompts[:-1]
        cache_entries, new_prompts = self.action_cache.read_action(prompts)
        if not new_prompts:
            return
        if self.retreive_all:
            yield from self._iter_step(cache_entries, new_prompts, stream_end)
        else:
            start_index = len(prompts) - len(new_prompts)
            for i in range(len(new_prompts) - 1):
                cache_entries, new_prompts = self.action_cache.read_action(prompts[:start_index+i+1])
                yield from self._iter_step(cache_entries, new_prompts, stream_end=False)
            cache_entries, new_prompts = self.action_cache.read_action(prompts)
            yield from self._iter_step(cache_entries, new_prompts, stream_end)


    def _iter_step(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str],
        stream_end: bool=False
    ) -> Generator[TextStreamer, None, None]:
        if stream_end:
            response_action = yield from self._iter_output(cache_entries, new_prompts)
            self.action_cache.write_action([response_action,])
        else:
            actions = []
            infer_action = yield from self._iter_inference(cache_entries, new_prompts)
            actions.append(infer_action)

            if self.hypothesize and infer_action.type == Wait:
                hypo_action = yield from self._iter_hypothesize(cache_entries, new_prompts)
                if hypo_action:
                    actions.append(hypo_action)

            self.action_cache.write_action(actions)

            if self.summarize_len > 0:
                new_entry = CacheEntry(actions=actions, prompts=new_prompts)
                new_entries = cache_entries + [new_entry] # add generated actions in this step for summarization
                num_non_wait = len([
                    action for cache_entry in new_entries 
                    for action in cache_entry.actions if action.type != Wait]
                )
                if num_non_wait >= self.summarize_len:
                    sum_action = yield from self._iter_summarize(new_entries, [])
                    if sum_action:
                        actions.append(sum_action)
                    if sum_action:
                        self.action_cache.reset_action([sum_action,])

    def _iter_inference(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> Generator[TextStreamer, None, Action]:
        msg = self.formatter.format_inference(cache_entries, new_prompts)
        response_gen = self.infer_model.stream(msg)
        text_streamer = TextStreamer(response_gen)
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
    ) -> Generator[TextStreamer, None, Action]:
        msg = self.formatter.format_output(cache_entries, new_prompts)
        if self.answer_format:
            msg[-1]['content'] += "\n\n"+self.answer_format
        response_gen = self.output_model.stream(msg)
        text_streamer = TextStreamer(response_gen)
        yield text_streamer
        response = text_streamer.text
        action = Action(type=Response, content=response)
        return action

    def _iter_hypothesize(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> Generator[TextStreamer, None, Action]:
        msg = self.formatter.format_hypothesize(cache_entries, new_prompts)
        response_gen = self.infer_model.stream(msg)
        text_streamer = TextStreamer(response_gen)
        yield text_streamer
        response = text_streamer.text
        action = self.formatter.parse_action(response, self.action_types)
        if not action:
            action = Action(type=Inference, content=response)
        return action

    def _iter_summarize(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> Generator[TextStreamer, None, Action|None]:
        msg = self.formatter.format_summarize(cache_entries, new_prompts)
        response_gen = self.infer_model.stream(msg)
        text_streamer = TextStreamer(response_gen)
        yield text_streamer
        response = text_streamer.text
        action = self.formatter.parse_action(response, self.action_types)
        return action


class CompleteCoTController(BaseController):
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
