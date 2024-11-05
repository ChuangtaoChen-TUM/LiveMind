""" Formats and templates for LMFormatter """
__all__  = [
    "FORMATTER_MAP",
    "LMFormat",
]

from enum import Enum
from ..action.abc import CacheEntry
from ..action.actions import Wait

class LMFormat(Enum):
    """ Format Enum for the LMFormatter """
    U_PI   = "user_prompt_inference"
    U_PLI  = "user_prompt_last_prompt_inference"
    U_SPI  = "user_sequence_prompt_inference"
    UA_PIL = "user_assistant_prompt_inference_last_prompt"
    UA_SPI = "user_assistant_sequence_prompt_inference"


def format_inference_sys() -> str:
    """The user is currently providing input, and you need to make inferences to obtain some temporary results based on the incomplete information available so far. These results will help you solve the problem more efficiently when more input arrives. You are given the incomplete problem and your previous inferences on the incomplete problem.

You can choose to make a new inference or wait if the incremental input is not enough to make a new inference.
If you choose to make a new inference, respond in the following format: 'action inference. {content}'.

The content should be relevant to content you are inferring based on the incomplete problem and your previous inferences.

If you choose to wait for more information, simply respond with 'action wait.' without any additional content."""
    return str(format_inference_sys.__doc__)


def format_output_sys() -> str:
    """You are given a problem and previous inferences you have made about the problem to solve the problem.

You should make the best use your previous inferences to answer to the problem.

You can answer directly if you can obtain the answer in your previous inferences, otherwise make additional inferences to solve the problem."""
    return str(format_output_sys.__doc__)


def format_u_pi(cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
    """Format the user prompt with user_prompt_inference format:
    User:
        Incomplete prompt: ...
        Previous inferences: (1) ... (2) ...
    """
    old_prompts = [prompt for entry in cache_entries for prompt in entry.prompts]
    prompts = old_prompts + new_prompts
    msg = []
    if prompts:
        msg.append("Incomplete prompt: ")
        msg.extend(prompts)

    infer_msgs = []
    index = 1
    old_actions = [action for entry in cache_entries for action in entry.actions]
    for action in old_actions:
        if action.formatted_content:
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

def format_u_pli(cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_prompt_last_prompt_inference format:
    User:
        Previous prompt: ...
        New prompt: ...
        Previous inferences: (1) ... (2) ...
    """
    i = len(cache_entries) - 1
    while i >= 0:
        if len(cache_entries[i].actions) == 1 and cache_entries[i].actions[0].type == Wait:
            i -= 1
        else:
            break

    old_prompts = [prompt for entry in cache_entries[:i+1] for prompt in entry.prompts]
    new_prompts = [prompt for entry in cache_entries[i+1:] for prompt in entry.prompts] + new_prompts
    old_actions = [action for entry in cache_entries for action in entry.actions]
    msg = []
    if old_prompts:
        msg.append("Previous prompt: ")
        msg.extend(old_prompts)
    infer_msgs = []
    index = 1
    for action in old_actions:
        if action.formatted_content:
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


def format_u_spi(cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_sequence_prompt_inference format:
    User: prompt: prompt 1 inference: inference 1 ... prompt: last_prompt
    """
    temp_prompts: list[str] = []
    msgs: list[str] = []

    for entry in cache_entries:
        prompts = entry.prompts
        infer_msg = " ".join(action.formatted_content for action in entry.actions if action.formatted_content)
        temp_prompts.extend(prompts)
        if infer_msg:
            if msgs:
                msgs.append("\n")
            msgs.append(f"Prompt: {''.join(temp_prompts)}")
            msgs.append(f"\nInference: {infer_msg}")
            temp_prompts = []

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


def format_ua_pil(cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
    """ Format the user prompt with user_assistant_prompt_inference_last_prompt format:
    User: previous prompt
    Assistant: previous inference
    User: new prompt
    """
    # regard last prompts with 'wait' actions as new prompts
    i = len(cache_entries) - 1
    while i >= 0:
        if len(cache_entries[i].actions) == 1 and cache_entries[i].actions[0].type == Wait:
            i -= 1
        else:
            break

    old_prompts = [prompt for entry in cache_entries[:i+1] for prompt in entry.prompts]
    new_prompts = [prompt for entry in cache_entries[i+1:] for prompt in entry.prompts] + new_prompts
    old_actions = [action for entry in cache_entries for action in entry.actions]

    msg = []
    infer_msgs = []
    for action in old_actions:
        if action.formatted_content:
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


def format_ua_spi(cache_entries: list[CacheEntry], new_prompts: list[str]) -> list[dict[str, str]]:
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
    msg_dict: list[dict] = []

    for entry in cache_entries:
        prompts = entry.prompts
        infer_msg = " ".join(action.formatted_content for action in entry.actions if action.formatted_content)
        temp_prompts.extend(prompts)
        if infer_msg:
            user_msg = {
                "role" : "user",
                "content" : "".join(temp_prompts)
            }
            msg_dict.append(user_msg)
            temp_prompts = []

            assistant_msg = {
                "role" : "assistant",
                "content" : infer_msg
            }
            msg_dict.append(assistant_msg)
            temp_prompts = []

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
    LMFormat.U_SPI : format_u_spi,
    LMFormat.UA_PIL : format_ua_pil,
    LMFormat.UA_SPI : format_ua_spi,
}
