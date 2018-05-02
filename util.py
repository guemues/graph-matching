import argparse
import os
import json
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
    hyperparamater_dict = {}

    for simulation_id, simulation_dict in infos.items():
        if simulation_dict['hyperparameter'] in hyperparamater_dict:
            hyperparamater_dict[simulation_dict['hyperparameter']].append(simulation_id)
        else:
            hyperparamater_dict[simulation_dict['hyperparameter']] = [simulation_id]

    total: DataFrame = pd.DataFrame()
    for hyperparameter, simulation_ids in hyperparamater_dict.items():
        for simulation_id in simulation_ids:
            _ = pd.read_csv(os.path.join(input_folder, str(simulation_id) + '.csv'))
            _['main_graph'] = str(simulation_id) + '_' + _['main_graph'].astype(str)
            total = total.append(find_tp_fp_fn(_))

    total = total.drop(['Unnamed: 0'], axis=1)

    total.to_csv(output_file, index=False)


if __name__ == '__main__':

    if args.hyperparameter:
        create_hyperparameter_result(args.input, args.output)
