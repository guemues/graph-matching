import argparse
import os
import json
from os.path import isfile, join

import pandas as pd
from pandas import DataFrame

from matching.statistics import find_tp_fp_fn

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--hyperparameter', dest='hyperparameter', type=bool, help='')
parser.add_argument('--output', dest='output', type=str, help='')
parser.add_argument('--input', dest='input', type=str, help='')

args = parser.parse_args()

info_file_str = os.path.join(args.input, 'info.json')
with open(info_file_str) as info_file:
    infos = json.load(info_file)


def create_hyperparameter_result(input_folder, output_file):
    total = pd.DataFrame()
    for _ in [f for f in os.listdir('./results/') if isfile(join('./results/', f)) and '.csv' in f]:
        a = pd.read_csv(os.path.join('./results/', _))
        a['main_graph'] = str(_) + '_' + a['main_graph'].astype(str)
        total = total.append(find_tp_fp_fn(a))

    total = total.drop(['Unnamed: 0'], axis=1)

    total.to_csv(output_file, index=False)


if __name__ == '__main__':

    if args.hyperparameter:
        create_hyperparameter_result(args.input, args.output)
