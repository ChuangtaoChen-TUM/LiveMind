""" action formatters: format the system and user prompts based on the action configurations and the performed actions."""
__all__ = [
    'LMFormatter',
    'CoTFormatter',
]

import re
import warnings
from collections.abc import Iterable
from .abc import (
    Action,
    ActionType,
    BaseActionFormatter,
)

class LMFormatter(BaseActionFormatter):
    """ Default action formatter for problem solving tasks with options
        If the arguments are changed, you need to clear the cached_prompts.
       """
    def __init__(
        self,
        action_types: Iterable[ActionType],
        append_new: bool=False,
        include_wait: bool=False
    ):
        self.action_types = action_types
        self.append_new = append_new
        self.include_wait = include_wait
        self.cached_prompts = {}
        if not append_new and include_wait:
            warnings.warn("include_wait is only effective when append_new is True.")

    def format_inference(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        sys_msg = self._format_inference_sys()
        user_msg = self._format_inference_user(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        return msg

    def format_output(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        sys_msg = self._format_output_sys()
        user_msg = self._format_output_user(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        return msg

    def _format_inference_sys(self) -> str:
        """ Format the system prompt:

            You are a helpful AI assistant. Your task is to understand and solve an incomplete problem. You are given {info_msg}.
            You can choose one of the following actions:
            
            action {name}: {instruction}
            action {name}: {instruction}
            ...

            Choose one of the above actions based on the problem and your previous actions. Your response must be formatted as: 'action {'name'/.../...}. {content}'.
            The content should be only relevant to the action you choose. Do not choose actions not listed above.

            {additional messages}

            If `append_new` is `False`, the info_msg is: "the incomplete problem and your previous actions on the incomplete problem".
            If `append_new` is `True`, the info_msg is: "the incomplete problem, your previous actions on the incomplete problem, and new content of the problem".
        """
        if "inference_sys" in self.cached_prompts:
            return self.cached_prompts["inference_sys"]
        msgs = []
        if self.append_new:
            info_msg = "You are given the incomplete problem, your previous actions on the incomplete problem, and new content of the problem. "
        else:
            info_msg = "You are given the incomplete problem and your previous actions on the incomplete problem."
        msgs.append("You are a helpful AI assistant. Your task is to understand and solve an incomplete problem.")
        msgs.append(info_msg)
        msgs.append("\nYou can choose one of the following actions:\n\n")
        add_msg = []
        for action_type in self.action_types:
            msgs.append(f"action {action_type.name}: {action_type.inst}\n")
            if action_type.add_msg:
                add_msg.append(action_type.add_msg)
        all_actions = "/".join([action_type.name for action_type in self.action_types])
        msgs.append("\nChoose one of the above actions based on the problem and your previous actions. ")
        msgs.append(f"Your response must be formatted as: 'action {{{all_actions}}}. {{content}}'.\n")
        msgs.append("The content should be only relevant to the action you choose. Do not choose actions not listed above.")
        if add_msg:
            msgs.append("\n\n")
            msgs.append("\n".join(add_msg))
        sys_msg = "".join(msgs)
        self.cached_prompts["inference_sys"] = sys_msg
        return sys_msg


    def _format_output_sys(self) -> str:
        """ Format the system prompt for final msg:

        You are a helpful AI assistant. Your are given a problem and previous inferences you have made about the problem. Your task is to provide the missing inferences and solve the problem. Answer directly if you have obtained the answer in your previous inferences, otherwise make minimal additional inferences to solve the problem.

        Using your previous inferences without directly mentioning them. For example, avoid using irrelevant phrases like "Based on my previous inferences" or "I can directly answer this question." Instead, respond directly with your new inferences and answer to the problem.
        """
        if "output_sys" in self.cached_prompts:
            return self.cached_prompts["output_sys"]
        sys_msg = "You are a helpful AI assistant. Your are given a problem and previous inferences you have made about the problem. "+\
                 "Your task is to provide the missing inferences and solve the problem. Answer directly if you have obtained the answer in your previous inferences, otherwise make minimal additional inferences to solve the problem.\n\n"+"Avoid using irrelevant phrases like \"Based on my previous inferences\" or \"I can directly answer this question.\" Instead, respond directly with your new inferences and answer to the problem."
        self.cached_prompts["output_sys"] = sys_msg
        return sys_msg


    def _format_inference_user(self, actions: list[Action], new_prompts: list[str]) -> str:
        """ Format the user prompt based on the actions and the new prompts:

            Incomplete problem: {old_prompt} {new_prompt}
            
            Your previous actions:
            (1) action {name}: {instruction}
            (2) action {name}: {instruction}

            If `append_new` is `True`, the new prompts will follow the previous actions instead of the old prompt:

            Incomplete problem: {old_prompt}
            
            Your previous actions:
            (1) action {name}: {instruction}
            (2) action {name}: {instruction}

            New content: {new_prompt}
            ...

            If `include_wait` is `True`, the prompts of previous wait actions will be moved to the new prompts.
        """
        if self.append_new and self.include_wait:
            i = len(actions) - 1
            while i >= 0 and actions[i].type.name == "wait":
                i -= 1
            old_prompts = [prompt for action in actions[:i+1] for prompt in action.prompts]
            wait_prompts = [prompt for action in actions[i+1:] for prompt in action.prompts]
            new_prompts = wait_prompts + new_prompts
        else:
            old_prompts = [prompt for action in actions for prompt in action.prompts]

        if self.append_new:
            prompt_msg = "".join(old_prompts)
        else:
            prompt_msg = "".join(old_prompts + new_prompts)
        msgs = []
        msgs.append(f"Incomplete problem: {prompt_msg}")
        action_msgs = []
        index = 1
        for action in actions:
            if (action_msg := action.formatted_content) is not None: # do not include actions with no content (e.g. wait)
                action_msgs.append(f"({index}) action {action.type.name}: {action_msg}")
                index += 1
        if action_msgs:
            msgs.append("\n\nYour previous actions:\n")
            msgs.append(" ".join(action_msgs))
        
        if self.append_new:
            msgs.append("\n\nNew content:")
            msgs.append("".join(new_prompts))
        return "".join(msgs)


    def _format_output_user(self, actions: list[Action], new_prompts: list[str]) -> str:
        """ Format the user prompt based on the actions and the new prompts:\n

            Your previous inferences:
            (1) action 'name': 'instruction'
            (2) action 'name': 'instruction'
            ...

            Complete problem: 'prompt'
        """
        old_prompts = [prompt for action in actions for prompt in action.prompts]
        prompt_msg = "".join(old_prompts + new_prompts)
        msgs = []
        action_msgs = []
        index = 0
        for action in actions:
            action_msg = action.formatted_content
            if action_msg is not None:
                action_msgs.append(f"({index}) action {action.type.name}: {action_msg}")
                index += 1
        if action_msgs:
            msgs.append("Your previous inferences:\n")
            msgs.append(" ".join(action_msgs))
        msgs.append(f"\n\nComplete problem: {prompt_msg}")
        return "".join(msgs)


    def parse_action(self, response: str) -> Action|None:
        """ Parse the user response to get the action. Return None if parsing fails.
            format: 'action {'name'/.../...}. {content}'
        """
        pattern = r"^action ([a-z]+).[ \n]*(.*)$"
        matched = re.match(pattern, response, re.DOTALL)
        if matched:
            action_name = matched.group(1)
            for action_type in self.action_types:
                if action_type.name == action_name:
                    content = matched.group(2)
                    return Action(type=action_type, content=content)
            return None
        else:
            return None


class CoTFormatter(BaseActionFormatter):
    """ The CompleteCoTFormatter is a formatter for the CompleteCoTController with CoT system prompt """
    def format_inference(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        raise NotImplementedError
    
    def format_output(self, history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
        sys_msg = self._format_output_sys()
        user_msg = self._format_output_user(history_actions, new_prompts)
        msg = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]
        return msg

    def _format_output_sys() -> str:
        """ Format the system prompt for final msg:

            You are a helpful AI assistant, and your tasks is to understand and solve a problem. Solve the problem by thinking step by step.
        """
        return "You are a helpful AI assistant, and your tasks is to understand and solve a problem. Solve the problem by thinking step by step."


    def _format_output_user(actions: list[Action], new_prompts: list[str]) -> str:
        """ Format the user prompt based on the actions and the new prompts:

            {old_prompt} {new_prompt}
        """
        old_prompts = [prompt for action in actions for prompt in action.prompts]
        return "".join(old_prompts + new_prompts)

    def parse_action(response: str) -> Action|None:
        """ Not implemented """
        raise NotImplementedError
