""" Base classes for the action type, action, action cache and action formatter. """
__all__ = [
    'ActionType',
    'Action',
]

from collections.abc import Callable
from dataclasses import dataclass, field



@dataclass
class Action:
    """ Action performed by the LLM. Can be 'inference', 'wait', 'hypothesize' and 'summarize'."""
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
    # how to display the action message in the prompt
    formatter: Callable[[str], str|None]=lambda x:x
