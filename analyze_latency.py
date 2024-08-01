import json
import argparse
import numpy
import glob
import os

def analyze_latency(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    len_dict = {}
    for entry in data:
        current_len = 0
        for step in entry['actions']:
            current_len += len(step) - 1
        if current_len not in len_dict:
            len_dict[current_len] = 0
        len_dict[current_len] += 1
    # for key in sorted(len_dict.keys()):
    #     print(f'len: {key}, count: {len_dict[key]}', end=', ')
    latencies = [entry['time_info']['latency'] for entry in data]
    correct = len([entry['correct'] for entry in data if entry['correct']])
    total_gen_times = [entry['time_info']['total_gen_time'] for entry in data]
    total = len(data)
    avg_latency = numpy.mean(numpy.array(latencies))
    avg_total_gen_time = numpy.mean(numpy.array(total_gen_times))
    print(f'file: {input_file}\tavg_latency: {avg_latency:.2f}, gen_time: {avg_total_gen_time:.2f}, correct: {correct*100/total:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+')
    args = parser.parse_args()
    files = []
    if len(args.input_files) == 1 and os.path.isdir(args.input_files[0]):
        files = glob.glob(os.path.join(args.input_files[0], '*.json'))
    else:
        for input_file in args.input_files:
            files += glob.glob(input_file)
    if not files:
        print('No files found')
    for file in files:
        if file.endswith('.json'):
            analyze_latency(file)
