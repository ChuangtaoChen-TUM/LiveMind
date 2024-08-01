""" Base classes for the action type, action, and cache entry. """
__all__ = [
    'ActionType',
    'Action',
]

from collections.abc import Callable
from dataclasses import dataclass

@dataclass(frozen=True)
class ActionType:
    """ Base class for the action type. """
    name: str
    inst: str
    # how to display the action message in the prompt
    formatter: Callable[[str], str|None]=lambda x:x


@dataclass
class Action:
    """ Action performed by the LLM. Can be 'inference', 'wait', 'hypothesize' and 'summarize'."""
    type: ActionType
    content: str=""

    @property
    def formatted_content(self) -> str|None:
        return self.type.formatter(self.content)
    
    def __repr__(self) -> str:
        return f"Action(type={self.type.name}, content={self.content})"


@dataclass
class CacheEntry:
    """ Cache entry for the action cache: contains a list of prompts and their corresponding actions. """
    actions: list[Action]
    prompts: list[str]
