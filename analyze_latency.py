import json
import argparse
import numpy

def analyze_latency(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    latencies = [entry['time_info']['latency'] for entry in data]
    avg_latency = numpy.mean(numpy.array(latencies))
    print(f'file: {input_file}, avg_latency: {avg_latency}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', required=True)
    args = parser.parse_args()
    for input_file in args.input_files:
        analyze_latency(input_file)
