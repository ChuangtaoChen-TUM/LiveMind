import json
import argparse
import glob
import os

GRANULARITY = ['char', 'word', 'clause', 'sentence']
def analyze_latency(input_file):
    print(f'file: {input_file}')
    with open(input_file, 'r') as f:
        data = json.load(f)
    sta_per_len = {}
    for entry in data:
        orig_len = len(entry['question'])
        if orig_len not in sta_per_len:
            sta_per_len[orig_len] = {
                'total': 0,
                'total_latency': 0,
                'total_gen_time': 0,
                'correct': 0,
            }
        sta_per_len[orig_len]['total'] += 1
        sta_per_len[orig_len]['total_latency'] += entry['time_info']['latency']
        sta_per_len[orig_len]['total_gen_time'] += entry['time_info']['total_gen_time']
        if entry['correct']:
            sta_per_len[orig_len]['correct'] += 1

    for key in sorted(sta_per_len.keys()):
        count = sta_per_len[key]['total']
        avg_latency = sta_per_len[key]['total_latency'] / count
        avg_gen_time = sta_per_len[key]['total_gen_time'] / count
        print(f'len {key},\tcount: {count},\tavg_latency: {avg_latency:.2f}\tavg_gen_time: {avg_gen_time:.2f}')


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
