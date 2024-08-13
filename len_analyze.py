import json
import argparse
import glob
import os
import csv

# u-pi, u-pli, ua-pil, u-spi, ua-spi
format_choices = ['u-pi', 'u-pli', 'ua-pil', 'u-spi', 'ua-spi']
segment_choices = ['sent', 'clause', 'word', 'char']

def sort_segment_fn(x):
    match x['segment']:
        case 'sent':
            return 0
        case 'clause':
            return 1
        case 'word':
            return 2
        case 'char':
            return 3
        case _:
            return 4

def get_choice(text: str, choices: list[str]):
    for c in choices:
        if c in text:
            return c
    return 'None'

def analyze_latency(input_files) -> tuple:
    gen_time_per_segment = {}
    file_infos = []
    for input_file in input_files:
        c_segment = get_choice(input_file, segment_choices)
        file_infos.append({'file': input_file, 'segment': c_segment})
    
    file_infos = sorted(file_infos, key=lambda x: sort_segment_fn(x))

    for file_info in file_infos:
        if file_info['segment'] not in gen_time_per_segment:
            gen_time_per_segment[file_info['segment']] = {}
        current_gen_time_per_len = gen_time_per_segment[file_info['segment']]

        input_file = file_info['file']
        with open(input_file, 'r') as f:
            data = json.load(f)
        for entry in data:
            orig_len = len(entry['question'])
            if orig_len not in current_gen_time_per_len:
                current_gen_time_per_len[orig_len] = []
            current_gen_time_per_len[orig_len].append(entry['time_info']['total_gen_time'])
    for seg in gen_time_per_segment.keys():
        for key in gen_time_per_segment[seg].keys():
            gen_time_per_segment[seg][key] = sum(gen_time_per_segment[seg][key])/len(gen_time_per_segment[seg][key])

    with open('latency.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["length"]+[key for key in gen_time_per_segment.keys()])
        keys = list(gen_time_per_segment.values())[0].keys()
        for key in keys:
            writer.writerow([key]+[gen_time_per_segment[segment][key] for segment in gen_time_per_segment.keys()])


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
    print(f"Analyzing latency for {len(files)} files")
    analyze_latency(files)
