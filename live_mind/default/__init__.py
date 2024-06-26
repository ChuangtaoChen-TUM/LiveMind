__all__ = [
    'LIVE_SYS_MSG',
    'FINAL_LIVE_MSG',
    'FULL_SYS_MSG', 
    'USE_COMP',
    'USE_OPTION',
    'SIMP_LIVE_SYS_MSG',
    'COMP_LIVE_SYS_MSG',
    'MMLU_FORMAT',
    'LLAMA_3_8B_PATH',
    'LLAMA_3_70B_PATH',
    'MMLU_PRO_PATH',
    'get_model'
]

from ..config import LLAMA_3_8B_PATH, LLAMA_3_70B_PATH, get_model, MMLU_PRO_PATH
SIMP_LIVE_SYS_MSG = """You are a helpful AI assistant. Your task is to understand and solve an incomplete problem. Based on the information provided and your previous actions, you can choose one of the following actions:

action inference: understand and make inferences based on the available information.
action wait: if you need more information or content, choose to wait.

Choose one of the above actions based on the problem and your previous actions. Your response must be formatted as: "action {inference/wait}. {content}".
The content should be as concise as possible and relevant to the action you choose. Do not choose actions that is not listed above.

If you choose action wait, simply respond with "action wait." without any additional content.
"""
COMP_LIVE_SYS_MSG = """You are a helpful AI assistant. Your task is to understand and solve an incomplete problem. Based on the information provided and your previous actions, you can choose one of the following actions:

action background: understand the topic background.
action inference: make inferences based on the available information.
action hypothesize: hypothesize what the final problem might be and attempt to solve it.
action wait: if you need more information or content, choose to wait.

Choose one of the above actions based on the problem and your previous actions. Your response must be formatted as: "action {background/inference/hypothesize/wait}. {content}".
The content should be as concise as possible and relevant to the action you choose. Do not choose actions that is not listed above.

If you choose action wait, simply respond with "action wait." without any additional content.
"""
USE_COMP = False
USE_OPTION = False
FINAL_LIVE_MSG_OPTION = "You are a helpful AI assistant. Answer directly if your answer in previous inferences is already in the options. If you need more inference to derive the final answer, think step by step based on your previous inferences."
FINAL_LIVE_MSG_NO_OPT = "You are a helpful AI assistant. Answer directly if your already know the answer. If you need more inference to derive the final answer, think step by step based on your previous inferences."
FULL_SYS_MSG = "You are a helpful AI assistant, and your tasks is to understand and solve a problem. Solve the problem by thinking step by step."

MMLU_FORMAT = "Your answer should end with 'The answer is (Choice)'."

def message_formatter(problem, actions: list[str], final_line=None):
    if final_line:
        action_msg = " ".join([f"({j+1}) {actions[j]}" for j in range(len(actions))])
        msg = f"Your previous inferences: {action_msg}"
        msg += f"\n\nComplete problem: {problem}\n\n{final_line}"
    else:
        msg = f"Incomplete problem: {problem}"
        if actions:
            action_msg = " ".join([f"({j+1}) {actions[j]}" for j in range(len(actions))])
            msg += f"\n\nYour previous actions: {action_msg}"
    return msg
