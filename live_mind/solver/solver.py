""" This module provides solvers with time tracking for single problems
For each problem, solver estimates the time to generate the text and measure the 
time of each response from the model using time.time(). Therefore, the generation 
time is related to the hardware and platform where the code is executed.

The solvers use the TextGenerator to generate the texts.
The generation strategy is "sentence".
The delay is estimated by a typing speed of 0.25 seconds per character.

The solver uses a model / assist_model to solve the problems.
The model should have the chat_complete method, which takes a list of messages and 
returns a list of responses.
Each response is in OpenAI's format, we need to use the:
    response["choices"][0]["message"]["content"] to get the response text.
    response["usage"] to get the token usages.
"""
import re
import logging
import time
from ..utils.text.text_generator import TextGenerator, build_delay_fn_char
from ..default import (
    message_formatter,
    FULL_SYS_MSG,
    COMP_LIVE_SYS_MSG,
    SIMP_LIVE_SYS_MSG,
    FINAL_LIVE_MSG_NO_OPT,
    FINAL_LIVE_MSG_OPTION
)
from .. import default

def solver(problem: str, model, answer_format:str, final_line:str, assist_model=None):
    """ solver for live_mind """
    if default.USE_COMP:
        live_sys_msg = COMP_LIVE_SYS_MSG
    else:
        live_sys_msg = SIMP_LIVE_SYS_MSG
    if default.USE_OPTION:
        final_sys_msg = FINAL_LIVE_MSG_OPTION
    else:
        final_sys_msg = FINAL_LIVE_MSG_NO_OPT
    delay_fn = build_delay_fn_char(0.25, 0)
    text_generator = TextGenerator(text=problem, strategy="sentence", delay_fn=delay_fn)
    history_action = []
    history_problem = ""
    usage = {
        "completion_tokens": [],
        "prompt_tokens": [],
        "total_tokens": []
    }
    current_time = 0
    total_wait_time = 0 # the arrival time of the last sentence
    total_gen_time = 0  # the total time for the model to generate the responses
    pattern = r"^action ([a-z]+).[ \n]*(.*)$"
    while not text_generator.empty():
        next_text, delay = text_generator.generate()
        total_wait_time += delay
        # wait for the next text if not arrived
        if current_time < total_wait_time:
            current_time = total_wait_time
        if not history_problem:
            history_problem = next_text
        else:
            history_problem = history_problem + " " + next_text

        query = message_formatter(history_problem, history_action)

        msg = [
            {"role": "system", "content": live_sys_msg},
            {"role": "user", "content": query}
        ]

        start = time.time()
        response = model.chat_complete([msg])[0]
        gen_time = time.time() - start

        # update curent time
        current_time += gen_time
        total_gen_time += gen_time

        text = response["choices"][0]["message"]["content"]
        usage["completion_tokens"].append(response["usage"]["completion_tokens"])
        usage["prompt_tokens"].append(response["usage"]["prompt_tokens"])
        usage["total_tokens"].append(response["usage"]["total_tokens"])
    
        matched = re.match(pattern, text, re.DOTALL)
        if matched:
            if matched.group(1) not in ["background", "inference", "hypothesize", "wait"]:
                logging.error(f"Invalid action: {matched.group(1)}")
            if matched.group(1) != "wait":
                history_action.append(text)
        else:
            logging.error(f"Invalid response: {text}")

    # the problem is complete, query the model with the complete problem
    if assist_model: # use the assist model for final completion if available
        model = assist_model

    total_wait_time += delay_fn(final_line)
    overhead_time = 0
    if current_time < total_wait_time:
        current_time = total_wait_time
    else:
        # this means the model has not finished the last response when the last sentence arrives
        overhead_time = current_time - total_wait_time

    if history_action:
        final_query = message_formatter(history_problem, history_action, final_line)
        msg = [
            {"role": "system", "content": final_sys_msg + " " + answer_format},
            {"role": "user", "content": final_query}
        ]
    else: # use conventional CoT if no actions are taken
        msg = [
            {"role": "system", "content": FULL_SYS_MSG + " " + answer_format},
            {"role": "user", "content": f"{history_problem}\n\n{final_line}"}
        ]

    start = time.time()
    response = model.chat_complete([msg])[0]
    gen_time = time.time() - start

    usage["completion_tokens"].append(response["usage"]["completion_tokens"])
    usage["prompt_tokens"].append(response["usage"]["prompt_tokens"])
    usage["total_tokens"].append(response["usage"]["total_tokens"])

    current_time += gen_time
    total_gen_time += gen_time

    # latency is defined as the time from the arrival of the last input sentence to the final response
    latency = current_time - total_wait_time
    time_info = {
        "current_time": current_time,
        "latency": latency,
        "total_gen_time": total_gen_time,
        "overhead_time": overhead_time
    }
    text = response["choices"][0]["message"]["content"]

    return text, usage, time_info


def solver_base(problem, model, answer_format:str, final_line, assist_model=None):
    """ baselien solver """
    delay_fn = build_delay_fn_char(0.25, 0)
    total_wait_time = 0
    total_wait_time += delay_fn(problem)
    total_wait_time += delay_fn(final_line)
    current_time = total_wait_time

    msg = [
        {"role": "system", "content": FULL_SYS_MSG + " " + answer_format},
        {"role": "user", "content": f"{problem}\n\n{final_line}"}
    ]

    start = time.time()
    response = model.chat_complete([msg])[0]
    gen_time = time.time() - start

    # the current time is the delay plus the generation time
    # the latency is the generation time
    current_time += gen_time
    latency = current_time - total_wait_time
    total_gen_time = gen_time

    time_info = {
        "current_time": current_time,
        "latency": latency,
        "total_gen_time": total_gen_time
    }

    usage = {
        "completion_tokens": [response["usage"]["completion_tokens"]],
        "prompt_tokens": [response["usage"]["prompt_tokens"]],
        "total_tokens": [response["usage"]["total_tokens"]]
    }

    text = response["choices"][0]["message"]["content"]

    return text, usage, time_info
