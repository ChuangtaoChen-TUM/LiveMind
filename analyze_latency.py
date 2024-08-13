import json
import argparse
import numpy
import glob
import os
import csv
import tabulate

def analyze_latency(input_file, exclude_one_line=False) -> tuple:
    with open(input_file, 'r') as f:
        data = json.load(f)
    if exclude_one_line:
        data = [entry for entry in data if len(entry['actions']) > 1]
    latencies = [entry['time_info']['latency'] for entry in data]
    correct = len([entry['correct'] for entry in data if entry['correct']])
    total_gen_times = [entry['time_info']['total_gen_time'] for entry in data]
    overhead_times = [entry['time_info']['overhead_time'] for entry in data]
    total = len(data)
    avg_latency = numpy.mean(numpy.array(latencies))
    avg_total_gen_time = numpy.mean(numpy.array(total_gen_times))
    avg_overhead_time = numpy.mean(numpy.array(overhead_times))

    file_base = os.path.basename(input_file)
    if total == 0:
        return (file_base, 0, 0, 0, 0)
    return (file_base, avg_latency, avg_total_gen_time, avg_overhead_time, correct*100/total)
    # print(f'file: {input_file}\tavg_latency: {avg_latency:.2f}\tgen_time: {avg_total_gen_time:.2f}\tcorrect: {correct*100/total:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+')
    parser.add_argument('--output', default=None)
    parser.add_argument("--exclude-one-line", action='store_true')
    args = parser.parse_args()
    output_file = args.output
    files = []
    results = []
    if len(args.input_files) == 1 and os.path.isdir(args.input_files[0]):
        files = glob.glob(os.path.join(args.input_files[0], '*.json'))
    else:
        for input_file in args.input_files:
            files += glob.glob(input_file)
    if not files:
        print('No files found')
        exit(1)
    for file in files:
        if file.endswith('.json'):
            results.append(analyze_latency(file, exclude_one_line=args.exclude_one_line))
    if not output_file:
        print(tabulate.tabulate(results, headers=['file', 'avg_latency', 'avg_gen_time', 'avg_overhead_time', 'correct']))
    else:
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'avg_latency', 'avg_gen_time', 'avg_overhead_time', 'correct'])
            for result in results:
                writer.writerow(result)
