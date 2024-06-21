""" This module contains the batch problem solvers
Batch solvers solve given problems in batch.
Batch solvers record the timestamps of each step and the token usages for each problem.
The outputs contain the final answers, timestamps, and the token usages for each completion.

The batch solvers use the BatchTextGenerator to generate the texts in batch.
The generation strategy is "sentence".
The delay is estimated by a typing speed of 0.25 seconds per character.

The batch solver uses a model / assist_model to solve the problems.
The model should have the chat_complete method, which takes a list of messages and 
returns a list of responses.
Each response is in OpenAI's format, we need to use the:
    response["choices"][0]["message"]["content"] to get the response text.
    response["usage"] to get the token usages.
"""
import re
import logging
from ..utils.text.text_generator import BatchTextGenerator, build_delay_fn_char
from ..default import (
    message_formatter,
    FULL_SYS_MSG,
    COMP_LIVE_SYS_MSG,
    SIMP_LIVE_SYS_MSG,
    FINAL_LIVE_MSG_OPTION,
    FINAL_LIVE_MSG_NO_OPT
)
from .. import default

def batch_solver(problems: list[str], model, answer_format: str, final_lines: list[str], assist_model=None):
    """ batch_solver for live_mind """
    if default.USE_COMP:
        live_sys_msg = COMP_LIVE_SYS_MSG
    else:
        live_sys_msg = SIMP_LIVE_SYS_MSG
    if default.USE_OPTION:
        final_sys_msg = FINAL_LIVE_MSG_OPTION
    else:
        final_sys_msg = FINAL_LIVE_MSG_NO_OPT
    assert len(problems) == len(final_lines), "The number of problems and final lines should be identical"
    delay_fn = build_delay_fn_char(0.25, 0)
    text_generator = BatchTextGenerator(texts=problems, strategy="sentence", delay_fn=delay_fn)
    num_problems = len(problems)

    # Initialize the tracking variables
    history_actions = [[] for _ in range(num_problems)]
    history_problems = [""] * num_problems
    timestamps = [[] for _ in range(num_problems)] # record the timestamps for each sentence
    usages = [
        {
            "completion_tokens": [],
            "prompt_tokens": [],
            "total_tokens": []
        } for _ in range(num_problems)
    ]

    # The pattern to match the action
    pattern = r"^action ([a-z]+).[ \n]*(.*)$"

    # get new texts, take an action for each new text
    while not all(text_generator.empty()):
        next_texts, delays = text_generator.generate()
        for i in range(num_problems): # only update the incomplete problems
            if next_texts[i]:
                timestamps[i].append(delays[i])
                if not history_problems[i]:
                    history_problems[i] = next_texts[i]
                else:
                    history_problems[i] = history_problems[i] + " " + next_texts[i]

        # else, query the model with the incomplete problems
        # the complete problems are paused until all problems are complete
        queries = []
        for i in range(num_problems):
            if next_texts[i]: # only query the incomplete problems
                query = message_formatter(history_problems[i], history_actions[i])
                queries.append(query)

        # the length of queries should be less or equal to the number of problems
        msgs = []
        # build messages for the model
        for query in queries:
            msgs.append([
                {"role": "system", "content": live_sys_msg},
                {"role": "user", "content": query}
            ])
        responses = model.chat_complete(msgs)
        # extract actions and usages from this step
        new_actions = []
        new_usages = []
        for response in responses:
            # record the usages
            new_usages.append(response["usage"])
            # record the actions
            text = response["choices"][0]["message"]["content"]
            matched = re.match(pattern, text, re.DOTALL)
            if matched:
                if matched.group(1) not in ["background", "inference", "hypothesize", "wait"]:
                    new_actions.append(None) # avoid insufficient actions to pop
                    logging.error(f"Invalid action: {matched.group(1)}")
                elif matched.group(1) == "wait":
                    new_actions.append(None) # do not record wait actions
                else:
                    new_actions.append(text)
            else:
                new_actions.append(None)
                logging.error(f"Invalid response: {text}")

        # update the history actions and usages to incomplete problems
        for i in range(num_problems):
            if next_texts[i]:
                next_action = new_actions.pop(0)
                if next_action is not None:
                    history_actions[i].append(next_action)
                next_usage = new_usages.pop(0)
                usages[i]["completion_tokens"].append(next_usage["completion_tokens"])
                usages[i]["prompt_tokens"].append(next_usage["prompt_tokens"])
                usages[i]["total_tokens"].append(next_usage["total_tokens"])

    # all problems are complete, query the model with the last lines
    if assist_model: # use the assist model for final completion if available
        model = assist_model

    # build messages for the model
    msgs = []
    for i in range(num_problems):
        if history_actions[i]:
            final_query = message_formatter(history_problems[i], history_actions[i], final_lines[i])
            msgs.append([
                {"role": "system", "content": final_sys_msg + " " + answer_format},
                {"role": "user", "content": final_query}
            ])
        else: # if no actions were taken, use the conventional CoT
            msgs.append([
                {"role": "system", "content": FULL_SYS_MSG + " " + answer_format},
                {"role": "user", "content": f"{history_problems[i]}\n\n{final_lines[i]}"}
            ])
        timestamps[i].append(delay_fn(final_lines[i]))

    responses = model.chat_complete(msgs)

    # update the usages for the final completions
    # the length of each usage should be identical to the number of timestamps, which is the
    # number of sentences of each problem
    for i in range(num_problems):
        next_usage = responses[i]["usage"]
        usages[i]["completion_tokens"].append(next_usage["completion_tokens"])
        usages[i]["prompt_tokens"].append(next_usage["prompt_tokens"])
        usages[i]["total_tokens"].append(next_usage["total_tokens"])

    texts = [response["choices"][0]["message"]["content"] for response in responses]
    infos = []
    for i in range(num_problems):
        infos.append({
            "action": history_actions[i],
            "timestamp": timestamps[i],
            "usage": usages[i]
        })

    return texts, infos


def batch_solver_base(problems, model, answer_format: str, assist_model=None, final_lines=None):
    """ baseline batch solver """
    delay_fn = build_delay_fn_char(0.25, 0)
    # Initialize the tracking variables
    num_problems = len(problems)
    timestamps = [[] for _ in range(num_problems)]
    usages = [
        {
            "completion_tokens": [],
            "prompt_tokens": [],
            "total_tokens": []
        } for _ in range(num_problems)
    ]

    # use delay_fn to estimate the delay for each problem
    for i in range(num_problems):
        timestamps[i].append(delay_fn(problems[i]))
        timestamps[i].append(delay_fn(final_lines[i]))

    # build messages for the model
    msgs = []
    for i in range(num_problems):
        msgs.append([
            {"role": "system", "content": FULL_SYS_MSG + " " + answer_format},
            {"role": "user", "content": problems[i]+"\n\n"+final_lines[i]}
        ])

    responses = model.chat_complete(msgs)
    # the length of each usage and timestamps should be 1, since the problem is complete
    for i, response in enumerate(responses):
        next_usage = response["usage"]
        usages[i]["completion_tokens"].append(next_usage["completion_tokens"])
        usages[i]["prompt_tokens"].append(next_usage["prompt_tokens"])
        usages[i]["total_tokens"].append(next_usage["total_tokens"])

    texts = [response["choices"][0]["message"]["content"] for response in responses]
    infos = []
    for i in range(num_problems):
        infos.append({
            "action": [],
            "usage": usages[i],
            "timestamp": timestamps[i]
        })
    return texts, infos
