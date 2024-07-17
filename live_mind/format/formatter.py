""" Formatters: Format prompts """

__all__ = [
    'LMFormatter',
    'CoTFormatter',
]

import re
from collections.abc import Iterable
from .functions import (
    LMFormat,
    format_inference_sys,
    format_output_sys,
    format_hypothesize_sys,
    format_summarize_sys,
    FORMATTER_MAP,
)
from .abc import BaseFormatter
from ..action.abc import (
    Action,
    ActionType,
    CacheEntry
)


class LMFormatter(BaseFormatter):
    """ Default action formatter for problem solving tasks with options
        If the arguments are changed, you need to clear the cached_prompts.
       """
    def __init__(
        self,
        format_type: LMFormat,
    ):
        self.cached_prompts: dict[str, str] = {}
        self.formatter_fn = FORMATTER_MAP[format_type]


    def format_inference(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        sys_msg = format_inference_sys()
        user_msg: list[dict[str, str]] = self.formatter_fn(cache_entries, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
        ]
        return msg


    def format_output(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        sys_msg = format_output_sys()
        user_msg = self.formatter_fn(cache_entries, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
        ]
        return msg


    def format_hypothesize(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        sys_msg = format_hypothesize_sys()
        user_msg = self.formatter_fn(cache_entries, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
        ]
        return msg


    def format_summarize(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
        sys_msg = format_summarize_sys()
        inferences = []
        for entry in cache_entries:
            for action in entry.actions:
                if action.formatted_content:
                    inferences.append(action.formatted_content)
        user_msg_dict = {
            "role": "user",
            "content": " ".join(inferences)
        }
        msg = [
            {"role": "system", "content": sys_msg},
            user_msg_dict
        ]
        return msg


    def parse_action(self, response: str, action_types: Iterable[ActionType]) -> Action|None:
        """ Parse the user response to get the action. Return None if parsing fails.
            format: 'action {'name'/.../...}. {content}'
        """
        pattern = r"^action ([a-z]+).[ \n]*(.*)$"
        matched = re.match(pattern, response, re.DOTALL)
        if matched:
            action_name = matched.group(1)
            for action_type in action_types:
                if action_type.name == action_name:
                    content = matched.group(2)
                    return Action(type=action_type, content=content)
            return None
        else:
            return None


class CoTFormatter(BaseFormatter):
    def __init__(self, use_cot: bool=False):
        self.use_cot = use_cot


    """ The CompleteCoTFormatter is a formatter for the CompleteCoTController with CoT system prompt """
    def format_output(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
        if self.use_cot:
            sys_msg = self._format_output_sys_cot()
        else:
            sys_msg = self._format_output_sys()
        user_msg = self._format_output_user(cache_entries, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        return msg


    def _format_output_sys(self) -> str:
        """You are a helpful AI assistant, and your tasks is to understand and solve a problem."""
        return str(self._format_output_sys.__doc__)


    def _format_output_sys_cot(self) -> str:
        """You are a helpful AI assistant, and your tasks is to understand and solve a problem. Solve the problem by thinking step by step."""
        return str(self._format_output_sys_cot.__doc__)


    def _format_output_user(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> str:
        """ Format the user prompt based on the actions and the new prompts:

            {old_prompt} {new_prompt}
        """
        old_prompts = [prompt for entry in cache_entries for prompt in entry.prompts]
        return "".join(old_prompts + new_prompts)


    # The baseline controller does not use the following methods
    def format_inference(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError


    def format_hypothesize(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError


    def format_summarize(self, cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError


    def parse_action(self, response: str, action_types: Iterable[ActionType]) -> Action|None:
        raise NotImplementedError
