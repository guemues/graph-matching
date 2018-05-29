import argparse
import os
import json
from os.path import isfile, join

import pandas as pd
from pandas import DataFrame

from matching.statistics import find_tp_fp_fn

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--output', dest='output', type=str, help='')
parser.add_argument('--input', dest='input', type=str, help='')

args = parser.parse_args()

info_file_str = os.path.join(args.input, 'info.json')
with open(info_file_str) as info_file:
    infos = json.load(info_file)


def combine_results(input_folder, output_file):
    total = pd.DataFrame()
    for _ in [f for f in os.listdir(input_folder) if isfile(join(input_folder, f)) and '.csv' in f]:
        a = pd.read_csv(os.path.join(input_folder, _))
        a['main_graph'] = str(_) + '_' + a['main_graph'].astype(str)
        total = total.append(find_tp_fp_fn(a))

    total.to_csv(output_file, index=False)


if __name__ == '__main__':
    combine_results(args.input, args.output)
