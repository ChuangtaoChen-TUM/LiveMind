import argparse
import json

def q_len(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    len_dict = {}

    for entry in data:
        c_len = len(entry['question'])
        if c_len not in len_dict:
            len_dict[c_len] = 0
        len_dict[c_len] += 1

    print('Length of questions and their counts:')    
    for key in sorted(len_dict.keys()):
        print(key, len_dict[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()
    q_len(args.input_file)
