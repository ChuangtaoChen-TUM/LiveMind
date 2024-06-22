from typing import List, Dict
from .action_manager import ActionManager

FINAL_SYS = """You are given your previous inferences on the user's prompt. Now you need to provide a final response based on the user's prompt and your previous inferences.

You must 1. keep your answer short by: answer directly if you have already obtained the answer;
2. try to use the results from your previous inferences more to minimize the additional inferences; do not repeat the same inferences again if they are already provided in your previous inferences.
2. use the results directly without mentioning "my previous inferences".
"""
SIMP_LIVE_SYS_MSG = """Your task is to understand and inference on an incomplete prompt from the user. Based on the information provided and your previous actions, you can choose one of the following actions:

action inference: understand and make inferences step by step based on the available information.
action wait: if you need more information or content for the inference, choose to wait.

Choose one of the above actions based on the problem and your previous actions. Your response must be formatted as: "action {inference/wait}. {content}".
The content should be relevant to the action you choose. Do not choose actions that is not listed above. The content should be only relevant to the action you choose. Do not include any additional meaningless sentences after the content such as "I will wait for more information...".

If you choose action wait, simply respond with "action wait." without any additional content.
"""
def form_prompt(
    user_message: str,
    dialogs: List[Dict[str, str]],
    action_manager: ActionManager,
    is_completed: bool = False,
):
    output_dialog = []
    actions, prompts = action_manager.read_action(user_message, save_prompts=False)
    # filter "" and None from actions
    actions = [action for action in actions if action]
    if is_completed:
        output_dialog.append(
            {
                "role": "system",
                "content": FINAL_SYS,
            }
        )
        for dialog in dialogs:
            output_dialog.append(dialog)
        if len(actions) > 0:
            action_msg = " ".join([f"({j+1}) {actions[j]}" for j in range(len(actions))])
            final_msg = f"Your previous inferences: {action_msg}\n\nUser prompt: {user_message}"
        else:
            final_msg = f"User prompt: {user_message}"
        output_dialog.append(
            {
                "role": "user",
                "content": final_msg,
            }
        )
        return output_dialog

    user_message = " ".join(prompts)
    output_dialog.append(
        {
            "role": "system",
            "content": SIMP_LIVE_SYS_MSG,
        }
    )
    final_msg = ""
    for dialog in dialogs:
        if dialog["role"] == "user":
            final_msg += f"User: {dialog['content']}\n"
        elif dialog["role"] == "assistant":
            final_msg += f"assistant: {dialog['content']}\n"
    if final_msg:
        final_msg = "History Dialog:\n" + final_msg
    if len(actions) > 0:
        action_msg = " ".join([f"({j+1}) {actions[j]}" for j in range(len(actions))])
        final_msg += f"Incomplete prompt from user: {user_message}\n\nYour previous actions: {action_msg}"
    else:
        final_msg += f"Incomplete prompt from user: {user_message}"
    output_dialog.append(
        {
            "role": "user",
            "content": final_msg,
        }
    )
    return output_dialog
