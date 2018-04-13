#!/usr/bin/env python
# This file contains pandas operations

import pandas as pd


def find_degree_counts(degrees_count):
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

    _ = _.reset_index()[['noise', 'threshold_ratio', 'degree', 'correctness', 'main_graph', 'index']]
    _ = _.groupby(['threshold_ratio', 'noise', 'correctness', 'main_graph', 'degree'])['index'].count().reset_index(
        ['correctness']).rename(columns={'index': 'count'})
    return _


def find_corrects_not_corrects(mapping_df_find_counts):
    _ = mapping_df_find_counts.copy()
    _ = _.loc[_['correctness'] == 'TRUE'].join(_.loc[_['correctness'] == 'FALSE'], rsuffix='_f', how='left')
    _['correctness_f'] = _['correctness_f'].fillna(value='FALSE')
    _['correctness'] = _['correctness'].fillna(value='TRUE')
    _[['count', 'count_f']] = _[['count', 'count_f']].fillna(value=0).astype(int)
    _ = _.rename(index=str, columns={"count": "corrects", "count_f": "not_corrects"})
    _ = _.drop(['correctness', 'correctness_f'], axis=1)

    return _


def find_sum(mappings_df_find_node_counts):
    _ = mappings_df_find_node_counts.copy()
    _ = _.groupby(['degree', 'threshold_ratio', 'noise'])['corrects', 'not_corrects', 'node_count'].sum()
    return _


def find_tp_fp_fn_tn(mapping_df_find_total_degree_node):
    _ = mapping_df_find_total_degree_node.copy()
    _['true_positive'] = _['corrects']
    _['false_positive'] = _['not_corrects']
    _['false_negative'] = _['node_count'] - _['corrects']
    _['true_negative'] = (_['node_count'] * (_['node_count'] - 1)) - _['not_corrects']
    _ = _[['true_positive', 'false_positive', 'false_negative', 'true_negative', 'node_count']]
    return _


def find_precision(mappings_w_tp_fp_fn_tn):
    _ = mappings_w_tp_fp_fn_tn.copy()
    _['precision'] = _['true_positive'] / (_['true_positive'] + _['false_negative'])
    return _


def find_sensitivity(mappings_w_tp_fp_fn_tn):
    _ = mappings_w_tp_fp_fn_tn.copy()
    _['sensitivity'] = _['true_positive'] / (_['true_positive'] + _['false_positive'])
    return _