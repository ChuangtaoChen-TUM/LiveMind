""" action formatters: format the system and user prompts based on the action configurations and the performed actions."""
__all__ = [
    'LMFormatter',
    'CoTFormatter',
]

import re
from collections.abc import Iterable
from .functions import (
    format_inference_sys,
    format_output_sys,
    format_hypothesize_sys,
    format_summarize_sys,
    FORMATTER_MAP
)
from .abc import BaseFormatter, LMFormat
from ..action.abc import (
    Action,
    ActionType
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
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        sys_msg = format_inference_sys()
        user_msg: list[dict[str, str]] = self.formatter_fn(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
        ]
        return msg


    def format_output(
        self,
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        sys_msg = format_output_sys()
        user_msg = self.formatter_fn(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
        ]
        return msg


    def format_hypothesize(
        self,
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        # TODO: Fix ua_spi format
        sys_msg = format_hypothesize_sys()
        user_msg = self.formatter_fn(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
        ]
        return msg


    def format_summarize(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        sys_msg = format_summarize_sys()
        user_msg = self.formatter_fn(history_actions, new_prompts) # use the same format as hypothesize
        msg = [
            {"role": "system", "content": sys_msg},
            *user_msg
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
    """ The CompleteCoTFormatter is a formatter for the CompleteCoTController with CoT system prompt """
    def format_output(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        sys_msg = self._format_output_sys()
        user_msg = self._format_output_user(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        return msg

    def _format_output_sys(self) -> str:
        """ Format the system prompt for final msg:

            You are a helpful AI assistant, and your tasks is to understand and solve a problem. Solve the problem by thinking step by step.
        """
        return "You are a helpful AI assistant, and your tasks is to understand and solve a problem."


    def _format_output_user(self, actions: list[Action], new_prompts: list[str]) -> str:
        """ Format the user prompt based on the actions and the new prompts:

            {old_prompt} {new_prompt}
        """
        old_prompts = [prompt for action in actions for prompt in action.prompts]
        return "".join(old_prompts + new_prompts)


    def format_inference(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError
    
    def format_hypothesize(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError
    
    def format_summarize(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError

    def parse_action(self, response: str, action_types: Iterable[ActionType]) -> Action|None:
        """ Not implemented """
        raise NotImplementedError
