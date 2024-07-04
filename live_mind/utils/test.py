import random
import string
from ..action.abc import ActionType

class DummyLLM:
    def __init__(self, action_types: list[ActionType], response_length: int=10) -> None:
        self.action_types = action_types
        self.response_length = response_length
    
    def chat_complete(self, messages: list[list[dict]]):
        batch_size = len(messages)
        responses = []
        for _ in range(batch_size):
            random_response = "".join(random.choices(string.ascii_lowercase, k=self.response_length))
            random_action = random.choice(self.action_types)
            action_msg = self.format_action(random_action)
            response = {
                "choices": [
                    {"message": {"content": action_msg+random_response}}
                ],
                "usage": {
                    "completion_tokens": 1,
                    "prompt_tokens": 2,
                    "total_tokens": 3
                }
            }
            responses.append(response)
        return responses

    def format_action(self, action: ActionType):
        return f"action {action.name}.\n"
