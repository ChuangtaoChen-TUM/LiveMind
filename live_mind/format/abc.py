__all__ = [
    "BaseFormatter",
    "LMFormat"
]

from abc import ABC, abstractmethod
from enum import Enum
from collections.abc import Iterable
from ..action.abc import Action, ActionType

class LMFormat(Enum):
    """ Enum for the LMFormatter """
    U_PI = "user_prompt_inference"
    U_PLI = "user_prompt_last_prompt_inference"
    U_PIL = "user_prompt_inference_last_prompt"
    U_IP = "user_inference_prompt"
    U_IPL = "user_inference_prompt_last_prompt"
    U_SPI = "user_sequence_prompt_inference"
    UA_PIL = "user_assistant_prompt_inference_last_prompt"
    UA_SPI = "user_assistant_sequence_prompt_inference"


class BaseFormatter(ABC):
    """ Base class for the action formatter.
    methods:
    - format_inference: format the prompts for inference stage
    - format_output: format the prompts for inference stage
    - parse_action: parse the action from the response
    """
    @abstractmethod
    def format_inference(
        self,
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def format_output(
        self,
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def format_hypothesize(
        self,
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass


    @abstractmethod
    def format_summarize(
        self,
        history_actions: list[Action],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass


    @abstractmethod
    def parse_action(
        self,
        response: str,
        action_types: Iterable[ActionType]
    ) -> Action|None:
        """ Read the action from the response. Return None if the parsing fails."""
        pass
