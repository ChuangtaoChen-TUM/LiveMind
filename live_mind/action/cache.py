""" Action cache based on the segmentation scheme. """
__all__ = ['SegmentActionCache']

from .abc import Action

class SegmentActionCache():
    """ Action cache based on the segmentation scheme.
        
        methods:
        - `read_action`: read the cached actions based on the prompt
        - `write_action`: write a new action to the cache
        - `clear_cache`: clear the cache
       """
    def __init__(
        self,
    ):
        self.cached_actions: list[Action] = []
        self.saved_prompts: list[str] = []
        self.saved_index = -1


    def read_action(
        self,
        prompts: list[str],
    ) -> tuple[list[Action], list[str]]:
        """ Read the action accoding to the prompt. If `drop_last` is True, the last segment is dropped when retrieving the actions.
        """

        min_len =  0
        # No stable prompts
        if len(prompts) <= min_len:
            return [], []

        a_index, p_index = self._retrieve_actions(prompts)
        actions = self.cached_actions[:a_index]
        # Record the index of the retrieved actions and new prompts
        new_prompts = prompts[p_index:]
        self.saved_prompts = new_prompts
        self.saved_index = a_index

        return actions, new_prompts


    def write_action(self, action: Action):
        """ write a new action with the complete prompt."""
        new_prompts = self.saved_prompts
        a_index = self.saved_index
        if a_index == -1:
            raise ValueError("Index error: call `read_action` before `write_action`.")
        self._write_action(action, new_prompts, a_index)
        self.saved_index = -1


    def clear_cache(self):
        """ Clear the cache."""
        self.cached_actions = []
        self.saved_prompts = []
        self.saved_index = -1


    def _write_action(self, action: Action, new_prompts: list[str], a_index: int):
        """ Write a new action with the action index and the new prompts."""
        if a_index < 0 or a_index > len(self.cached_actions):
            raise ValueError("Invalid index to write action")
        # update the action
        action.prompts = new_prompts
        if a_index == len(self.cached_actions):
            self.cached_actions.append(action)
        elif self.cached_actions[a_index] != action:
            self.cached_actions[a_index] = action
            self.cached_actions = self.cached_actions[: a_index + 1]


    def _retrieve_actions(self, prompts: list[str]) -> tuple[int, int]:
        """ Retrieve the actions by index based on the prompts."""
        # initialize the index
        prompt_len = len(prompts)
        p_index = 0
        a_index = 0
        # Retrieve cached actions
        for action in self.cached_actions:
            prompts_of_action = action.prompts
            if prompt_len - p_index < len(prompts_of_action):
                # The prompts of the action is longer than the remaining prompts
                break
            if prompts[p_index: p_index + len(prompts_of_action)] == prompts_of_action:
                p_index += len(prompts_of_action)
                a_index += 1
            else:
                # The prompts of the action do not match the given prompts
                break
        return a_index, p_index

