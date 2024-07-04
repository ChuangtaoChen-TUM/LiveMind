""" Base classes for the action type, action, action cache and action formatter. """
__all__ = [
    'ActionType',
    'Action',
    'BaseActionCache',
    'BaseActionFormatter'
]

from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from dataclasses import dataclass, field


@dataclass
class Action:
    """ Action performed by the LLM."""
    type: 'ActionType'
    content: str=""
    prompts: list[str]=field(default_factory=list)

    @property
    def formatted_content(self) -> str|None:
        return self.type.formatter(self.content)
    
    def __repr__(self) -> str:
        return f"Action(type={self.type.name}, content={self.content}, {len(self.prompts)} prompts)"


@dataclass(frozen=True)
class ActionType:
    """ Definition of an action."""
    name: str
    inst: str
    # additional message for the action
    add_msg: str|None=None
    # how to display the action message in the prompt
    formatter: Callable[[str], str|None]=lambda x:x
    modifier: Callable[[Iterable[Action], str], list[Action]]=lambda x,y:x # modify previous actions based on the content


class BaseActionCache(ABC):
    """ Base class for the action cache.
        methods:
        - read_action: read the cached actions based on the prompt
        - write_action: write a new action to the cache
        - clear_cache: clear the cache
    """
    @abstractmethod
    def read_action(self, prompt: str) -> tuple[list[Action], list[str]]:
        pass

    @abstractmethod
    def write_action(self, action: Action, prompt: str|None):
        pass

    @abstractmethod
    def clear_cache(self):
        pass


class BaseActionFormatter(ABC):
    """ Base class for the action formatter.
    methods:
    - sys_formatter: format the system prompt based on action configurations
    - prompt_formatter: format the prompt based on the performed actions and the prompt
    - final_sys_formatter: format the final system prompt based on action configurations
    - final_prompt_formatter: format the final prompt based on the performed actions and the prompt
    - parse_action: parse the action from the response
    """
    @abstractmethod
    def sys_formatter(self, action_types: Iterable[ActionType]) -> str:
        pass

    @abstractmethod
    def prompt_formatter(self, actions: list[Action], new_prompts: list[str]) -> str:
        pass

    @abstractmethod
    def final_sys_formatter(self, action_types: Iterable[ActionType]) -> str:
        pass

    @abstractmethod
    def final_prompt_formatter(self, actions: list[Action], new_prompts: list[str]) -> str:
        pass

    @abstractmethod
    def parse_action(self, response: str, action_types: Iterable[ActionType]) -> Action|None:
        """ Read the action from the response. Return None if the parsing fails."""
        pass
