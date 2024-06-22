from typing import TypedDict, List
import nltk

class Action(TypedDict):
    prompt: str
    action: str

class ActionManager:
    def __init__(self):
        self.cached_actions: List[Action] = []
        self.saved_prompts = []

    def read_action(self, prompt, save_prompts=False):
        prompts = nltk.sent_tokenize(prompt)
        if len(prompts) <= 1:
            return [], None

        stable_prompts = prompts[:-1]
        actions = []
        for i in range(len(stable_prompts)):
            if i < len(self.cached_actions) and stable_prompts[i] == self.cached_actions[i]["prompt"]:
                actions.append(self.cached_actions[i]["action"])
            else:
                break
        while len(actions) > 0 and actions[-1] is None:
            actions.pop()
        if save_prompts:
            self.saved_prompts = stable_prompts
        return actions, stable_prompts

    def write_action(self, action):
        prompts = self.saved_prompts
        for i in range(len(prompts)):
            if i < len(self.cached_actions):
                if i == len(prompts) - 1:
                    self.cached_actions[i]["prompt"] = prompts[i]
                    self.cached_actions[i]["action"] = action
                    self.cached_actions = self.cached_actions[: i + 1]
                elif prompts[i] != self.cached_actions[i]["prompt"]:
                    self.cached_actions[i]["action"] = None
                    self.cached_actions = self.cached_actions[: i + 1]
            else:
                if i == len(prompts) - 1:
                    self.cached_actions.append({"prompt": prompts[i], "action": action})
                else:
                    self.cached_actions.append({"prompt": prompts[i], "action": None})
