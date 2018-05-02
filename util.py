import os
import json
import pandas as pd
from pandas import DataFrame

from matching.statistics import find_tp_fp_fn

info_file_str = './results/info.json'
with open(info_file_str) as info_file:
    infos = json.load(info_file)

hyperparamater_dict = {}

for simulation_id, simulation_dict in infos.items():
    if simulation_dict['hyperparameter'] in hyperparamater_dict:
        hyperparamater_dict[simulation_dict['hyperparameter']].append(simulation_id)
    else:
        hyperparamater_dict[simulation_dict['hyperparameter']] = [simulation_id]

total: DataFrame = pd.DataFrame()
for hyperparameter, simulation_ids in hyperparamater_dict.items():
    for simulation_id in simulation_ids:
        _ = pd.read_csv(os.path.join('./results/', str(simulation_id) + '.csv'))
        _['main_graph'] = str(simulation_id) + '_' + _['main_graph'].astype(str)
        _['hyperparameter'] = str(hyperparameter)
        total = total.append(find_tp_fp_fn(_))

total = total.drop(['Unnamed: 0'], axis=1)

total.to_csv('results.csv', index=False)


