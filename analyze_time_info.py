""" python mmlu_pro_analyze_time_info.py ./output/time_info/.../all.json, this will create a report in the same directory as the input file """
import json
import numpy as np
import csv
import argparse
import os
from live_mind.utils.dataset.mmlu_pro import get_prediction
from live_mind.utils.dataset import sent_len

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

total_latency =[[] for _ in range(num_files)]
total_acc = [{"correct": 0, "count": 0} for _ in range(num_files)]

acc_per_category = [{} for _ in range(num_files)]
latency_per_category = [{} for _ in range(num_files)]
acc_per_len = [{} for _ in range(num_files)]
latency_per_len = [{} for _ in range(num_files)]

for i in range(num_files):
    for entry in data[i]:
        category = entry["category"]
        current_len = sent_len(entry["question"]) + 1
        if category not in acc_per_category[i]:
            acc_per_category[i][category] = {"correct": 0, "count": 0}
            latency_per_category[i][category] = []
        if current_len not in acc_per_len[i]:
            acc_per_len[i][current_len] = {"correct": 0, "count": 0}
            latency_per_len[i][current_len] = []
        if entry["answer"] == get_prediction(entry["solution"], verbose=False, guess=False):
            total_acc[i]["correct"] += 1
            acc_per_category[i][category]["correct"] += 1
            acc_per_len[i][current_len]["correct"] += 1
        total_acc[i]["count"] += 1
        acc_per_category[i][category]["count"] += 1
        acc_per_len[i][current_len]["count"] += 1
        
        total_latency[i].append(entry["time_info"]["latency"])
        latency_per_category[i][category].append(entry["time_info"]["latency"])
        latency_per_len[i][current_len].append(entry["time_info"]["latency"])

len_for_each_category = {}
for category in acc_per_category[0]:
    len_for_each_category[category] = acc_per_category[0][category]["count"]

for i in range(1, num_files):
    for category in acc_per_category[i]:
        if acc_per_category[i].keys() != len_for_each_category.keys():
            raise ValueError("Categories are not consistent across files")
        if len_for_each_category[category] != acc_per_category[i][category]["count"]:
            raise ValueError("Counts are not consistent across files")

len_for_each_len = {}
for length in acc_per_len[0]:
    len_for_each_len[length] = acc_per_len[0][length]["count"]

for i in range(1, num_files):
    for length in acc_per_len[i]:
        if acc_per_len[i].keys() != len_for_each_len.keys():
            raise ValueError("Lengths are not consistent across files")
        if len_for_each_len[length] != acc_per_len[i][length]["count"]:
            raise ValueError("Counts are not consistent across files")


with open(output_dir+'/timeinfo_by_category.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    row_title = ["Category", "Total"] + [f"File {i+1} Accuracy" for i in range(num_files)] + [f"File {i+1} Latency" for i in range(num_files)]
    writer.writerow(row_title)
    row_all = ["All", len(data[0])] + [100*total_acc[i]["correct"]/total_acc[i]["count"] for i in range(num_files)] + [np.mean(total_latency[i]) for i in range(num_files)]
    writer.writerow(row_all)
    categories = list(acc_per_category[0].keys())
    for category in categories:
        total = acc_per_category[0][category]["count"]
        row = [category, total] + \
            [100*acc_per_category[i][category]["correct"]/total for i in range(num_files)] +\
            [np.mean(latency_per_category[i][category]) for i in range(num_files)]
        writer.writerow(row)

# Open the CSV file in write mode
with open(output_dir+'/timeinfo_by_len.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    row_title = ["Length", "Total"] + [f"File {i+1} Accuracy" for i in range(num_files)] + [f"File {i+1} Latency" for i in range(num_files)]
    writer.writerow(row_title)
    row_all = ["All", len(data[0])] + [100*total_acc[i]["correct"]/total_acc[i]["count"] for i in range(num_files)] + [np.mean(total_latency[i]) for i in range(num_files)]
    writer.writerow(row_all)
    lengths = list(acc_per_len[0].keys())
    for length in lengths:
        total = acc_per_len[0][length]["count"]
        row = [length, total] + \
            [100*acc_per_len[i][length]["correct"]/total for i in range(num_files)] +\
            [np.mean(latency_per_len[i][length]) for i in range(num_files)]
        writer.writerow(row)
