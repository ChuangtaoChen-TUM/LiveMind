__all__  = [
    "FORMATTER_MAP",
]

from .abc import LMFormat
from ..action import Action
from ..action.actions import Wait

def format_inference_sys() -> str:
    """You are a helpful large language model tasked with solving a problem based on user input. The user is currently providing input, and you need to make inferences based on the incomplete information available so far. These inferences will help you solve the problem more efficiently when more input arrives. You are given the incomplete problem and your previous inferences on the incomplete problem.

You can choose to make a new inference or wait if you need more information.
If you choose to make a new inference, respond in the following format: 'action inference. {content}'.
The content should be only relevant to content you are inferring based on the incomplete problem and your previous inferences.

If you choose to wait, respond with 'action wait.' without any additional content."""
    return str(format_inference_sys.__doc__)


def format_output_sys() -> str:
    """You are a helpful AI assistant. Your are given a problem and previous inferences you have made about the problem. Your task is to make inferences to solve the problem.

You should use your previous inferences without directly mentioning them. For example, avoid using irrelevant phrases like "Based on my previous inferences". Instead, respond directly with your new inferences and answer to the problem. Answer directly if you have obtained the answer in your previous inferences, otherwise make minimal additional inferences to solve the problem."""
    return str(format_output_sys.__doc__)


def format_hypothesize_sys() -> str:
    """You are a helpful large language model tasked with solving a problem based on user input. The user is currently providing input, and you need to hypothesize what the user might input next and make new inferences based on your hypothesis. You are given the incomplete problem and your previous inferences on the incomplete problem.

If the hypothesized input is the final problem, try to solve the hypothesized problem based on the current information. Otherwise make inferences based on the hypothesized input.

You should respond in the following format: 'action hypothesize. {content}'."""
    return str(format_hypothesize_sys.__doc__)


def format_summarize_sys() -> str:
    """You are a helpful AI assistant. Your task is to summarize your previous inferences you have made so far. Provide a concise summary of the inferences you have made while retaining the important information. You are given the incomplete problem and your previous inferences on the incomplete problem.
    
Respond in the following format: 'action summarize. {summarization}'."""
    return str(format_summarize_sys.__doc__)

def format_u_pi(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """Format the user prompt with user_prompt_inference format:
    User:
        Incomplete prompt: ...
        Previous inferences: (1) ... (2) ...
    """
    old_prompts = [prompt for action in history_actions for prompt in action.prompts]
    prompts = old_prompts + new_prompts
    msg = []
    if prompts:
        msg.append("Incomplete prompt: ")
        msg.extend(prompts)

    infer_msgs = []
    index = 1
    for action in history_actions:
        if action.type != Wait and action.formatted_content:
            infer_msgs.append(f"({index}) {action.formatted_content}")
            index += 1
    if infer_msgs:
        if msg:
            msg.append("\n")
        msg.append("Previous inferences: ")
        msg.extend(" ".join(infer_msgs))
    
    msg_dict = {
        "role" : "user",
        "content" : "".join(msg)
    }
    return [msg_dict]


def format_u_pli(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_prompt_last_prompt_inference format:
    User:
        Previous prompt: ...
        New prompt: ...
        Previous inferences: (1) ... (2) ...
    """
    # regard last prompts with 'wait' actions as new prompts
    i = len(history_actions) - 1
    while i >= 0 and history_actions[i].type == Wait:
        i -= 1
    old_prompts = [prompt for action in history_actions[:i+1] for prompt in action.prompts]
    wait_prompts = [prompt for action in history_actions[i+1:] for prompt in action.prompts]
    new_prompts = wait_prompts + new_prompts

    msg = []
    if old_prompts:
        msg.append("Previous prompt: ")
        msg.extend(old_prompts)
    infer_msgs = []
    index = 1
    for action in history_actions:
        if action.type != Wait and action.formatted_content:
            infer_msgs.append(f"({index}) {action.formatted_content}")
            index += 1
    if new_prompts:
        if msg:
            msg.append("\n")
        msg.append("New prompt: ")
        msg.extend(new_prompts)
    if infer_msgs:
        if msg:
            msg.append("\n")
        msg.append("Previous inferences: ")
        msg.extend(" ".join(infer_msgs))

    msg_dict = {
        "role" : "user",
        "content" : "".join(msg)
    }
    return [msg_dict]

def format_u_pil(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_prompt_inference_last_prompt format:
    User:
        Previous prompt: ...
        Previous inferences: (1) ... (2) ...
        New prompt: ...
    """
    # regard last prompts with 'wait' actions as new prompts
    i = len(history_actions) - 1
    while i >= 0 and history_actions[i].type == Wait:
        i -= 1
    old_prompts = [prompt for action in history_actions[:i+1] for prompt in action.prompts]
    wait_prompts = [prompt for action in history_actions[i+1:] for prompt in action.prompts]
    new_prompts = wait_prompts + new_prompts

    msg = []
    if old_prompts:
        msg.append("Previous prompt: ")
        msg.extend(old_prompts)
    infer_msgs = []
    index = 1
    for action in history_actions:
        if action.type != Wait and action.formatted_content:
            infer_msgs.append(f"({index}) {action.formatted_content}")
            index += 1
    if infer_msgs:
        if msg:
            msg.append("\n")
        msg.append("Previous inferences: ")
        msg.extend(" ".join(infer_msgs))
    if new_prompts:
        if msg:
            msg.append("\n")
        msg.append("New prompt: ")
        msg.extend(new_prompts)

    msg_dict = {
        "role" : "user",
        "content" : "".join(msg)
    }
    return [msg_dict]


def format_u_ip(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_inference_prompt format:
    User:
        Previous inferences: (1) ... (2) ...
        Incomplete prompt: ...
    """
    old_prompts = [prompt for action in history_actions for prompt in action.prompts]
    msg = []
    infer_msgs = []
    index = 1
    for action in history_actions:
        if action.type != Wait and action.formatted_content:
            infer_msgs.append(f"({index}) {action.formatted_content}")
            index += 1
    if infer_msgs:
        msg.append("Previous inferences: ")
        msg.extend(" ".join(infer_msgs))
    if old_prompts:
        if msg:
            msg.append("\n")
        msg.append("Incomplete prompt: ")
        msg.extend(old_prompts)
    msg.extend(new_prompts)
    
    msg_dict = {
        "role" : "user",
        "content" : "".join(msg)
    }
    return [msg_dict]

def format_u_ipl(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_inference_prompt_last_prompt format:
    User:
        Previous inferences: (1) ... (2) ...
        Previous prompt: ...
        New prompt: ...
    """
    # regard last prompts with 'wait' actions as new prompts
    i = len(history_actions) - 1
    while i >= 0 and history_actions[i].type == Wait:
        i -= 1
    old_prompts = [prompt for action in history_actions[:i+1] for prompt in action.prompts]
    wait_prompts = [prompt for action in history_actions[i+1:] for prompt in action.prompts]
    new_prompts = wait_prompts + new_prompts

    msg = []
    infer_msgs = []
    index = 1
    for action in history_actions:
        if action.type != Wait and action.formatted_content:
            infer_msgs.append(f"({index}) {action.formatted_content}")
            index += 1
    if infer_msgs:
        msg.append("Previous inferences: ")
        msg.extend(" ".join(infer_msgs))
    if old_prompts:
        if msg:
            msg.append("\n")
        msg.append("Previous prompt: ")
        msg.extend(old_prompts)
    if new_prompts:
        if msg:
            msg.append("\n")
        msg.append("New prompt: ")
        msg.extend(new_prompts)

    msg_dict = {
        "role" : "user",
        "content" : "".join(msg)
    }
    return [msg_dict]

def format_u_spi(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_sequence_prompt_inference format:
    User: prompt: prompt 1 inference: inference 1 ... prompt: last_prompt
    """
    temp_prompts: list[str] = []
    temp_infer_msgs: list[str] = []
    msgs: list[str] = []
    for action in history_actions:
        has_prompts = bool(action.prompts)
        not_wait = action.type != Wait

        # (has_prompts and not_wait) can be `True, True` (inference)`, 
        # `True, False` (wait), `False, True` (hypothesize, summarize)
        # it can never be `False, False` (invalid action)
        assert has_prompts or not_wait

        if has_prompts and temp_infer_msgs:
            # if current action is inference or wait, flush previous actions
            if msgs:
                msgs.append("\n")
            msgs.append("Inference: ")
            msgs.append(" ".join(temp_infer_msgs))
            temp_infer_msgs = []

        # save current action prompts
        temp_prompts.extend(action.prompts)

        # if current action is not wait, flush the prompts
        if not_wait:
            if msgs:
                msgs.append("\n")
            msgs.append("Prompt: ")
            msgs.append("".join(temp_prompts))
            temp_prompts = []

        # save current action content
        if action.formatted_content:
            temp_infer_msgs.append(action.formatted_content)


    # flush the last action
    if temp_infer_msgs:
        if msgs:
            msgs.append("\n")
        msgs.append("Inference: ")
        msgs.append(" ".join(temp_infer_msgs))

    new_prompts = temp_prompts + new_prompts
    assert new_prompts
    if msgs:
        msgs.append("\n")
    msgs.append("Prompt: ")
    msgs.append("".join(new_prompts))
    msg_dict = {
        "role" : "user",
        "content" : "".join(msgs)
    }
    return [msg_dict]


def format_ua_pil(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_assistant_prompt_inference_last_prompt format:
    User: previous prompt
    Assistant: previous inference
    User: new prompt
    """
    # regard last prompts with 'wait' actions as new prompts
    i = len(history_actions) - 1
    while i >= 0 and history_actions[i].type == Wait:
        i -= 1
    old_prompts = [prompt for action in history_actions[:i+1] for prompt in action.prompts]
    wait_prompts = [prompt for action in history_actions[i+1:] for prompt in action.prompts]
    new_prompts = wait_prompts + new_prompts

    msg = []
    infer_msgs = []
    for action in history_actions:
        if action.type != Wait and action.formatted_content:
            infer_msgs.append(action.formatted_content)

    
    has_old = bool(old_prompts)
    has_infer = bool(infer_msgs)
    has_new = bool(new_prompts)

    if has_old:
        user_msg = {
            "role" : "user",
            "content" : "".join(old_prompts)
        }
        msg.append(user_msg)

        assert has_infer
        assistant_msg = {
            "role" : "assistant",
            "content" : " ".join(infer_msgs)
        }
        msg.append(assistant_msg)

        assert has_new
        user_msg = {
            "role" : "user",
            "content" : "".join(new_prompts)
        }
        msg.append(user_msg)
    else:
        assert not has_infer # if there is no old prompts, there should be no inferences
        assert has_new
        all_prompts = old_prompts + new_prompts
        user_msg = {
            "role" : "user",
            "content" : "".join(all_prompts)
        }
        msg.append(user_msg)
    return msg


def format_ua_spi(history_actions: list[Action], new_prompts: list[str]) -> list[dict[str, str]]:
    """Format the user prompt with user_assistant_sequence_prompt_inference format:
    User: prompt 1
    Assistant: inference 1
    User: prompt 2
    Assistant: inference 2
    ...
    User: Last prompt

    If the prompts correspond to an wait action, they are merged to further prompts.
    If the action has no prompts (e.g. hypothesis, summarize), they are merged to the previous action.
    """
    temp_prompts: list[str] = []
    temp_infer_msgs: list[str] = []
    msg_dict: list[dict] = []
    for action in history_actions:
        has_prompts = bool(action.prompts)
        not_wait = action.type != Wait

        # (has_prompts and not_wait) can be `True, True` (inference)`, 
        # `True, False` (wait), `False, True` (hypothesize, summarize)
        # it can never be `False, False` (invalid action)
        assert has_prompts or not_wait

        if has_prompts and temp_infer_msgs:
            # if current action is inference or wait, flush previous actions
            assistant_msg = {
                "role" : "assistant",
                "content" : "action inference. " + " ".join(temp_infer_msgs)
            }
            msg_dict.append(assistant_msg)
            temp_infer_msgs = []

        # save current action prompts
        temp_prompts.extend(action.prompts)

        # if current action is not wait, flush the prompts
        if not_wait:
            user_msg = {
                "role" : "user",
                "content" : "".join(temp_prompts)
            }
            msg_dict.append(user_msg)
            temp_prompts = []
        
        # save current action content
        if action.formatted_content:
            temp_infer_msgs.append(action.formatted_content)


    # flush the last action
    if temp_infer_msgs:
        assistant_msg = {
            "role" : "assistant",
            "content" : "action inference. " + " ".join(temp_infer_msgs)
        }
        msg_dict.append(assistant_msg)

    new_prompts = temp_prompts + new_prompts
    assert new_prompts
    user_msg = {
        "role" : "user",
        "content" : "".join(new_prompts)
    }
    msg_dict.append(user_msg)

    return msg_dict

FORMATTER_MAP = {
    LMFormat.U_PI : format_u_pi,
    LMFormat.U_PLI : format_u_pli,
    LMFormat.U_PIL : format_u_pil,
    LMFormat.U_IP : format_u_ip,
    LMFormat.U_IPL : format_u_ipl,
    LMFormat.U_SPI : format_u_spi,
    LMFormat.UA_PIL : format_ua_pil,
    LMFormat.UA_SPI : format_ua_spi
}
