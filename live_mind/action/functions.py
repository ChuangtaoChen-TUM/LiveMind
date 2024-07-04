""" Functions for actions. """
__all__ = [
    'none_fn',
    'modifier_summarize',
]

from .abc import Action

def none_fn(*args, **kwargs):
    """ A dummy function that does nothing. """
    return None

def modifier_summarize(actions: list[Action], content: str) -> list[Action]:
    """ Summrize previous actions. Delete all actions except the last one and add the prompts to the last action """
    last_action = actions[-1]
    prompts = [prompt for action in actions for prompt in action.prompts]
    last_action.prompts = prompts
    actions = [last_action, ]
