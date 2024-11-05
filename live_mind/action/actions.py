""" This module contains the actions defined for the live_mind framework. """
__all__ = [
    'Inference',
    'Wait',
    'Response',
]

from .abc import ActionType

Inference = ActionType(
    name="inference",
    inst="understand and make inferences based on the available information."
)

Wait = ActionType(
    name="wait",
    inst="if you need more information or content, choose to wait.",
)

Response = ActionType(
    name="response",
    inst="response to the user's prompt based on the available information.",
)
