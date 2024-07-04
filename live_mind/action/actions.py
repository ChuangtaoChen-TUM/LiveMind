""" This module contains the actions defined for the live_mind framework. """
__all__ = [
    'Infer',
    'Wait',
    'Response',
    'Background',
    'Hypothesize',
    'Summarize'
]

from .abc import ActionType
from .functions import modifier_summarize, none_fn

Infer = ActionType(
    name="inference",
    inst="understand and make inferences based on the available information."
)

Wait = ActionType(
    name="wait",
    inst="if you need more information or content, choose to wait.",
    add_msg="If you choose action wait, simply respond with \"action wait.\" without any additional content.",
    formatter=none_fn # do not display the action in the prompt
)

""" The special `response` action type will be used for multi-turn dialogues. Previous responses are stored 
as historical `response` actions. """
Response = ActionType(
    name="response",
    inst="response to the user's prompt based on the available information.",
)

Background = ActionType(
    name="background",
    inst="understand the topic background."
)

Hypothesize = ActionType(
    name="hypothesize",
    inst="hypothesize what the final problem might be and attempt to solve it."
)

Summarize = ActionType(
    name="summarize",
    inst="summarize your previous actions and the current state of the problem.",
    modifier=modifier_summarize
)
