from collections.abc import Callable, Generator
import logging
from .abc import BaseController, BaseModel
from .action import Action
from .action.cache import SegmentActionCache
from .action.actions import Wait, Inference, Hypothesize, Summarize
from .format import BaseFormatter

class LMController(BaseController):
    """ the LMController maintains a cache of actions and generates prompts based on current prompt and cached actions.

    If the actions correspond to the prompt are in the cache, the controller returns `None`,
    otherwise, it will generate the prompts based on the actions and the prompt.
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
        """ Generate the prompts based on the user input and the cached actions """
        prompts = self.segmenter(prompt)
        if not stream_end:
            prompts = prompts[:-1]
        _, new_prompts = self.action_cache.read_action(prompts)
        if not new_prompts:
            return
        if self.retreive_all:
            yield from self._step(prompts, stream_end)
        else:
            start_index = len(prompts) - len(new_prompts)
            for i in range(len(new_prompts) - 1):
                yield from self._step(prompts[:start_index+i+1], stream_end=False)
            yield from self._step(prompts, stream_end)


    def _step(self, prompts:list[str], stream_end: bool=False):
        if stream_end:
            yield self._output(prompts)
        else:
            action, response = self._inference(prompts)
            yield response
            if self.hypothesize and isinstance(action, Action) and action.type == Wait:
                yield self._hypothesize(prompts)
            if self.summarize_len > 0:
                summarization = self._summarize(prompts)
                if summarization:
                    yield summarization


    def _inference(self, prompts: list[str]) -> tuple[Action|None, str]:
        """ Generate the prompts for the inference stage """
        actions, new_prompts = self.action_cache.read_action(prompts)
        msg = self.formatter.format_inference(actions, new_prompts)
        response = self.infer_model.chat_complete(msg)
        # if the action is not parsed, write a wait as a placeholder to avoid frequent inference
        action = self._update(response, force_write=True)
        return action, response


    def _output(self, prompts: list[str]):
        actions, new_prompts = self.action_cache.read_action(prompts)
        msg = self.formatter.format_output(actions, new_prompts)
        if self.answer_format:
            msg[-1]['content'] += "\n\n"+self.answer_format
        response = self.output_model.chat_complete(msg)
        return response


    def _hypothesize(self, prompts: list[str]):
        """ Generate the prompts for the hypothesis stage """
        actions, new_prompts = self.action_cache.read_action(prompts)
        msg = self.formatter.format_hypothesize(actions, new_prompts)
        response = self.infer_model.chat_complete(msg)
        self._update(response)
        return response


    def _summarize(self, prompts: list[str]):
        """ Generate the prompts for the summary stage """
        actions, new_prompts = self.action_cache.read_action(prompts)
        num_non_wait = len([action for action in actions if action.type != Wait])
        if num_non_wait < self.summarize_len:
            return ""
        msg = self.formatter.format_summarize(actions, new_prompts)
        response = self.infer_model.chat_complete(msg)
        action = self.formatter.parse_action(response, self.action_types)
        if isinstance(action, Action) and action.type == Summarize:
            prompts = [prompt for action in self.action_cache.cached_actions for prompt in action.prompts]
            action.prompts = prompts
            self.action_cache.clear_cache()
            self.action_cache.cached_actions = [action,]
        return response


    def _update(self, response:str, force_write:bool=False) -> Action|None:
        """ Update the controller with the response from the LLM, if the response contains an action, write the action to the cache """
        action = self.formatter.parse_action(response, self.action_types)
        if action is not None:
            self.action_cache.write_action(action)
        else:
            if force_write:
                self.action_cache.write_action(Action(type=Wait))
            if self.logger:
                self.logger.warning(f"Action is not parsed from the response: {response}")
        return action


    def reset(self):
        """ Reset the controller for a new conversation """
        self.action_cache.clear_cache()


class CompleteCoTController(BaseController):
    """ The CompleteCoTController is a controller that simulate conventional conversation with the LLM with complete prompts. """
    def __init__(self, formatter: BaseFormatter, output_model: BaseModel) -> None:
        self.formatter = formatter
        self.output_model = output_model

    def __call__(self, prompt:str, stream_end:bool=False) -> Generator[str, None, None]:
        """ Only generate the prompts when the stream ends """
        if not stream_end:
            return
        msg = self.formatter.format_output([], [prompt,])
        yield self.output_model.chat_complete(msg)

    def reset(self):
        pass
