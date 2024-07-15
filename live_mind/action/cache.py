""" Action cache based on the segmentation scheme. """
__all__ = ['SegmentActionCache']

from .abc import CacheEntry, Action

class SegmentActionCache():
    """ Action cache based on the segmentation scheme.

        methods:
        - `read_action`: read the cached actions and new prompt based on the prompt
        - `write_action`: write new actions to the cache
        - `clear_cache`: clear the cache
       """
    def __init__(
        self,
    ):
        self.cached_actions: list[CacheEntry] = []
        self.saved_prompts: list[str] = []
        self.saved_index = -1


    def read_action(
        self,
        prompts: list[str],
    ) -> tuple[list[CacheEntry], list[str]]:
        """ Read the cached entries and new prompts based on the prompts. """

        if len(prompts) == 0:
            return [], []

        e_index, p_index = self._get_index(prompts)
        retrieved_entries = self.cached_actions[:e_index]
        new_prompts = prompts[p_index:]
        # Record the index of the retrieved entries and new prompts
        self.saved_prompts = new_prompts
        self.saved_index = e_index

        return retrieved_entries, new_prompts


    def write_action(self, actions: list[Action]):
        """ Write new actions to the cache, must be called after `read_action` """
        new_prompts = self.saved_prompts
        e_index = self.saved_index
        if e_index == -1:
            raise ValueError("Index error: call `read_action` before `write_action`.")
        new_entry = CacheEntry(actions, new_prompts)
        self._write_action(new_entry, e_index)
        self.saved_index = -1


    def clear_cache(self):
        """ Clear the cache """
        self.cached_actions = []
        self.saved_prompts = []
        self.saved_index = -1


    def _write_action(self, new_entry: CacheEntry, e_index: int):
        """ Write a new entry with the action index to the cache """
        if e_index < 0 or e_index > len(self.cached_actions):
            raise ValueError("Invalid index to write action")
        # update the action
        if e_index == len(self.cached_actions):
            # append the new entry
            self.cached_actions.append(new_entry)
        elif self.cached_actions[e_index] != new_entry:
            # update the entry
            self.cached_actions[e_index] = new_entry
            self.cached_actions = self.cached_actions[: e_index + 1]


    def _get_index(self, prompts: list[str]) -> tuple[int, int]:
        """ get the index of entry and prompt based on the prompts.
        `e_index` is the index of the first entry whose prompts do not match the input prompts.
        `p_index` is the index of the first prompt that does not match the cached prompts.

        If all entries match the prompts, `e_index` is the length of the cached actions.
        If all prompts match the cached prompts, `p_index` is the length of the prompts.
        """
        # initialize the index
        prompt_len = len(prompts)
        p_index = 0
        e_index = 0
        # Retrieve cached actions
        for action in self.cached_actions:
            prompts_of_action = action.prompts
            if prompt_len - p_index < len(prompts_of_action):
                # The prompts of the action is longer than the remaining prompts
                break
            if prompts[p_index: p_index + len(prompts_of_action)] == prompts_of_action:
                p_index += len(prompts_of_action)
                e_index += 1
            else:
                # The prompts of the action do not match the given prompts
                break
        return e_index, p_index
