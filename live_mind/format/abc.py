""" Abstract base class for formatter """

__all__ = [
    "BaseFormatter",
]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from ..action.abc import Action, ActionType, CacheEntry


class BaseFormatter(ABC):
    """ Base class for the action formatter.
    methods:
    - format_inference: format the prompts for inference stage
    - format_output: format the prompts for inference stage
    - format_hypothesize: format the prompts for hypothesis stage
    - format_summarize: format the prompts for summary stage
    - parse_action: parse the action from the model's response
    """
    @abstractmethod
    def format_inference(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def format_output(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def format_hypothesize(
        self,
        cache_entries: list[CacheEntry],
        new_prompts: list[str]
    ) -> list[dict[str, str]]:
        pass


    @abstractmethod
    def format_summarize(
        self,
        cache_entries: list[CacheEntry],
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
