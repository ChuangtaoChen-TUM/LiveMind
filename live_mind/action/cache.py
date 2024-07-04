""" Action cache based on the segmentation scheme. """
__all__ = ['SegmentActionCache']

from collections.abc import Callable
from .abc import Action, BaseActionCache

class SegmentActionCache(BaseActionCache):
    """ Action cache based on the segmentation scheme.
        
        args:
        - `segmenter`: segmenter function
        - `save_prompts`: whether to save the prompts to avoid re-segmentation in writing actions when reading actions
        - `drop_last`: whether to drop the last segment when reading actions
        
        methods:
        - `read_action`: read the cached actions based on the prompt
        - `write_action`: write a new action to the cache
        - `clear_cache`: clear the cache
       """
    def __init__(
        self,
        segmenter:Callable[[str], list[str]],
        save_prompts:bool=True,
        drop_last:bool=True
    ):
        self.segmenter = segmenter
        self.cached_actions: list[Action] = []
        self.save_prompts = save_prompts
        self.drop_last = drop_last
        self.saved_prompt = None
        self.saved_prompts: list[str] = []
        self.saved_index = -1


    def read_action(
        self,
        prompt: str,
    ) -> tuple[list[Action], list[str]]:
        """ Read the action accoding to the prompt. If `drop_last` is True, the last segment is dropped when retrieving the actions.
        """
        if self.segmenter is None:
            raise ValueError("segmenter is not set.")
        prompts = self.segmenter(prompt)

        min_len = 1 if self.drop_last else 0
        # No stable prompts
        if len(prompts) <= min_len:
            return [], []

        if self.drop_last:
            stable_prompts = prompts[:-1]
        else:
            stable_prompts = prompts

        a_index, p_index = self._retrieve_actions(stable_prompts)
        actions = self.cached_actions[:a_index]
        # Record the index of the retrieved actions and new prompts
        new_prompts = stable_prompts[p_index:]
        if self.save_prompts:
            self.saved_prompts = new_prompts
            self.saved_index = a_index

        return actions, new_prompts


    def write_action(self, action: Action, prompt: str|None):
        """ write a new action with the complete prompt."""
        if self.save_prompts and self.saved_prompt == prompt:
            new_prompts = self.saved_prompts
            a_index = self.saved_index
        else:
            if prompt is None:
                raise ValueError("Prompt is not set.")
            prompts = self.segmenter(prompt)
            min_len = 1 if self.drop_last else 0
            if len(prompts) <= min_len:
                raise ValueError("Invalid prompt")
            if self.drop_last:
                stable_prompts = prompts[:-1]
            else:
                stable_prompts = prompts
            a_index, p_index = self._retrieve_actions(stable_prompts)
            new_prompts = stable_prompts[p_index:]
        self._write_action(action, new_prompts, a_index)


    def clear_cache(self):
        """ Clear the cache."""
        self.cached_actions = []
        self.saved_prompts = []
        self.saved_index = -1


    def _write_action(self, action: Action, new_prompts: list[str], a_index: int):
        """ Write a new action with the action index and the new prompts."""
        if not new_prompts:
            raise ValueError("No prompts are saved.")
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

