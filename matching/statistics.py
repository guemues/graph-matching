#!/usr/bin/env python
# This file contains pandas operations

import pandas as pd
import numpy as np
import itertools

def find_degree_counts(degree_count):
    #                   node_count
    # main_graph degree
    # 0          162             1
    #            139             2
    #            132             1
    #            131             1
    #            130             1

    _ = pd.DataFrame(columns=['degree', 'node_count'])

    for degree, count in degree_count.items():
        _ = _.append({
            'degree': int(degree),
            'node_count': int(count)}, ignore_index=True)
    return _.set_index(['degree'])



def find_degree_counts_(degrees_count):
    #                   node_count
    # main_graph degree
    # 0          162             1
    #            139             2
    #            132             1
    #            131             1
    #            130             1

    _ = pd.DataFrame(columns=['main_graph', 'degree', 'node_count'])

    for network_id, degree_count in enumerate(degrees_count):
        for degree, count in degree_count.items():
            _ = _.append({
                'main_graph': int(network_id),
                'degree': int(degree),
                'node_count': int(count)}, ignore_index=True)
    return _.set_index(['main_graph', 'degree'])


def find_counts(mapping_df_with_node_degree_count):
    #                                correctness  node_count_1  \
    # threshold_ratio noise degree_1
    # 0.005           0.0   10             FALSE           540
    #                       10              TRUE          7065
    #                       11             FALSE           270
    #                       11              TRUE          5805
    #                       12             FALSE           180
    #
    #                                 greater_degree_count_1
    # threshold_ratio noise degree_1
    # 0.005           0.0   10                           540
    #                       10                          7065
    #                       11                           270
    #                       11                          5805
    #                       12                           180
    _ = mapping_df_with_node_degree_count.copy()
    _ = _.rename(index=str, columns={'node_degree_1': 'degree'})

    _ = _.reset_index()[['noise', 'hyperparameter', 'threshold_ratio', 'degree', 'correctness', 'index']]
    _ = _.groupby(['threshold_ratio', 'noise', 'hyperparameter', 'correctness', 'degree'])['index'].count().reset_index().rename(columns={'index': 'count'})

    _[['threshold_ratio', 'noise', 'hyperparameter', 'degree']] = _[['threshold_ratio', 'noise', 'hyperparameter', 'degree']].astype(str)
    return _


def find_counts_(mapping_df_with_node_degree_count):
    #                                correctness  node_count_1  \
    # threshold_ratio noise degree_1
    # 0.005           0.0   10             FALSE           540
    #                       10              TRUE          7065
    #                       11             FALSE           270
    #                       11              TRUE          5805
    #                       12             FALSE           180
    #
    #                                 greater_degree_count_1
    # threshold_ratio noise degree_1
    # 0.005           0.0   10                           540
    #                       10                          7065
    #                       11                           270
    #                       11                          5805
    #                       12                           180
    _ = mapping_df_with_node_degree_count.copy()
    _ = _.rename(index=str, columns={'node_degree_1': 'degree'})

    _ = _.reset_index()[['noise', 'threshold_ratio', 'degree', 'correctness', 'main_graph', 'index']]
    _ = _.groupby(['threshold_ratio', 'noise', 'correctness', 'main_graph', 'degree'])['index'].count().reset_index(
        ['correctness']).rename(columns={'index': 'count'})
    return _


def fill_empty(mapping_df_find_counts, thresholds, noises, hyperparameters, degrees):
    all_possible_index = [i for i in itertools.product(thresholds, noises, hyperparameters, degrees, ['TRUE', 'FALSE'])]
    np_all_possible_index = np.array(all_possible_index)

    empty_df = pd.DataFrame(data={'threshold_ratio': np_all_possible_index[:, 0],
                                  'noise': np_all_possible_index[:, 1],
                                  'hyperparameter': np_all_possible_index[:, 2],
                                  'degree': np_all_possible_index[:, 3],
                                  'correctness': np_all_possible_index[:, 4], 'count': [0] * len(np_all_possible_index)})
    _ = empty_df.join(
        mapping_df_find_counts.reset_index().set_index(['threshold_ratio', 'noise', 'hyperparameter', 'degree', 'correctness']),
        on=['threshold_ratio', 'noise', 'hyperparameter', 'degree', 'correctness'], lsuffix='l')
    _ = _.fillna(0)
    return _.drop(['countl'], axis=1).set_index(['threshold_ratio', 'noise', 'hyperparameter', 'degree'])


def find_corrects_not_corrects(mapping_df_find_counts):
    _ = mapping_df_find_counts.copy().set_index(['threshold_ratio', 'noise', 'hyperparameter', 'degree'])
    _ = _.loc[_['correctness'] == 'TRUE'].join(_.loc[_['correctness'] == 'FALSE'], rsuffix='_f', how='left')
    _ = _.rename(index=str, columns={"count": "corrects", "count_f": "not_corrects"})
    _ = _.drop(['correctness', 'correctness_f', 'main_graph_f'], axis=1)

    return _.reset_index()


def find_node_counts(mappings_df_find_corrects_not_corrects , degrees_counts):
    _ :pd.DataFrame= mappings_df_find_corrects_not_corrects.copy()
    degrees_counts[['node_count']] = degrees_counts[['node_count']].astype(int)

    _['degree'] = _['degree'].astype(int)
    _ = _.set_index(['degree'])
    _ = degrees_counts.join(_)
    return _.reset_index()


def find_sum(mappings_df_find_node_counts):
    _ = mappings_df_find_node_counts.copy()
    _ = _.groupby(['degree', 'threshold_ratio', 'noise'])['corrects', 'not_corrects', 'node_count'].sum()
    return _


def find_tp_fp_fn(mapping_df_find_total_degree_node):
    _ = mapping_df_find_total_degree_node.copy()
    _['true_positive'] = _['corrects']
    _['false_positive'] = _['not_corrects']
    _['false_negative'] = _['node_count'] - _['corrects']
    return _


def find_precision(mappings_w_tp_fp_fn_tn):
    _ = mappings_w_tp_fp_fn_tn.copy()
    _['precision'] = _['true_positive'] / (_['true_positive'] + _['false_negative'])
    return _


def find_sensitivity(mappings_w_tp_fp_fn_tn):
    _ = mappings_w_tp_fp_fn_tn.copy()
    _['sensitivity'] = _['true_positive'] / (_['true_positive'] + _['false_positive'])
    return _