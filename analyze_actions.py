""" python ./mmlu_pro_analyze_actions.py ./output/batched/.../all.json, this will create a report in the same directory as the input file """
import json
import argparse
import os
import re
import csv

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="+")
args = parser.parse_args()

files = args.files
data = []
for file in files:
    with open(file, "r", encoding='utf-8') as file:
        data.append(json.load(file))

num_files = len(files)
if num_files == 1:
    output_dir = os.path.dirname(files[0])
else:
    output_dir = "./"

actions = [[] for _ in range(num_files)]
pattern = r"^action (\w+)."
action_types = []

for i in range(num_files):
    for entry in data[i]:
        action_for_entry = []
        action_steps = len(entry["timestamp"]) - 1
        valid_actions = len(entry["action"])
        num_wait_actions = action_steps - valid_actions
        completion_tokens = entry["usage"]["completion_tokens"]
        assert completion_tokens.count(4) == num_wait_actions
        start_pos = 0
        for j in range(action_steps):
            if completion_tokens[j] == 4:
                action = "wait"
            else:
                matched = re.match(pattern, entry["action"][start_pos])
                if matched:
                    action = matched.group(1)
                else:
                    action = "unknown"
                start_pos += 1
            action_for_entry.append(action)
            if action not in action_types:
                action_types.append(action)
        actions[i].append(action_for_entry)

actions_per_len = [{} for _ in range(num_files)]
actions_per_step = [{} for _ in range(num_files)]

for i in range(num_files):
    for action_per_entry in actions[i]:
        action_len = len(action_per_entry)
        if action_len not in actions_per_len[i]:
            actions_per_len[i][action_len] = []
        actions_per_len[i][action_len].extend(action_per_entry)
        for j in range(action_len):
            if j+1 not in actions_per_step[i]:
                actions_per_step[i][j+1] = []
            actions_per_step[i][j+1].append(action_per_entry[j+1])

with open(os.path.join(output_dir, "actions_per_len.csv"), "w", encoding='utf-8') as file:
    writer = csv.writer(file)
    row_title = ["length"] + action_types
    writer.writerow(row_title)
    for i in range(num_files):
        for length, actions in actions_per_len[i].items():
            row = [length] + [actions.count(action)/len(actions) for action in action_types]
            writer.writerow(row)

with open(os.path.join(output_dir, "actions_per_step.csv"), "w", encoding='utf-8') as file:
    writer = csv.writer(file)
    row_title = ["step"] + action_types
    writer.writerow(row_title)
    for i in range(num_files):
        for step, actions in actions_per_step[i].items():
            row = [step] + [actions.count(action)/len(actions) for action in action_types]
            writer.writerow(row)
