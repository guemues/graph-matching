#!/usr/bin/env python
# This file contains pandas operations

import pandas as pd


def find_correctness(mapping_df):
    _ = mapping_df.copy()
    _['correctness']=_.apply(lambda x: 'TRUE' if x['node_1'] == x['node_2'] else  'FALSE', axis=1)
    return _

def find_node_degrees(graph):
    #       degree
    # node
    # 0        106
    # 1         13
    # 2         71
    # 3        104
    # 4         26

    nodes, degrees = [], []
    for i, v in graph.degree:
        nodes.append(i)
        degrees.append(v)
    df = pd.DataFrame(
    data={
        "node": nodes,
        "degree": degrees
    })
    return df.set_index('node')


def find_degree_counts(node_degrees):
    #         node  greater_degree_count
    # degree
    # 10       157                  1000
    # 11       129                   843

    _ = node_degrees.copy()
    _ = _.reset_index().groupby('degree').count().astype(int)
    _['greater_degree_count'] = _.loc[::-1, 'node_count'].cumsum()[::-1]
    return _


def find_mapping_df_with_node_degree(mapping_df, node_degrees):
    #    index  node_1  node_2  noise  threshold_ratio correctness  degree_1  \
    # 0      0       0       0    0.0            0.005        TRUE       106
    # 1      1       1       1    0.0            0.005        TRUE        13
    # 2      2       2       2    0.0            0.005        TRUE        71
    # 3      3       3       3    0.0            0.005        TRUE       104
    # 4      4       4       4    0.0            0.005        TRUE        26
    #
    #    degree_2
    # 0       106
    # 1        13
    # 2        71
    # 3       104
    # 4        26
    _ = mapping_df.copy()
    _ = _.join(node_degrees, on='node_1').rename(columns={'degree': 'degree_1'}).join(node_degrees, on='node_2').rename(columns={'degree': 'degree_2'})
    return _


def find_mapping_df_with_node_degree_count(mapping_df_node_degrees, degree_counts):
    #    index  node_1  node_2  noise  threshold_ratio correctness  degree_1  \
    # 0      0       0       0    0.0            0.005        TRUE       106
    # 1      1       1       1    0.0            0.005        TRUE        13
    # 2      2       2       2    0.0            0.005        TRUE        71
    # 3      3       3       3    0.0            0.005        TRUE       104
    # 4      4       4       4    0.0            0.005        TRUE        26
    #
    #    degree_2  node_count_1  greater_degree_count_1  node_count_2  \
    # 0       106             2                      10             2
    # 1        13            79                     607            79
    # 2        71             2                      26             2
    # 3       104             1                      12             1
    # 4        26             6                     156             6
    #
    #    greater_degree_count_2
    # 0                      10
    # 1                     607
    # 2                      26
    # 3                      12
    # 4                     156
    _ = mapping_df_node_degrees.copy()
    _ = _.join(degree_counts, on='degree_1').rename(columns={'node_count': 'node_count_1', 'greater_degree_count': 'greater_degree_count_1'}).join(degree_counts, on='degree_2').rename(columns={'node_count': 'node_count_2', 'greater_degree_count': 'greater_degree_count_2'})
    return _


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
    _ = _.groupby(['threshold_ratio', 'noise', 'degree_1', 'correctness'])[
        'node_count_1'].count().reset_index(['correctness'])
    return _


def find_total_degree_node(mapping_df_counts, comp_size):
    _ = mapping_df_counts.copy()
    _['total_degree_node'] = (_['node_count_1'] * comp_size).astype(int)
    _['total_greater_degree_count'] = (_['greater_degree_count_1'] * comp_size).astype(int)
    _.drop(['node_count_1', 'greater_degree_count_1'], axis=1)
    return _

def find_corrects_not_corrects(mapping_df_with_counts):
    _ = mapping_df_with_counts.copy()
    _ = _.loc[_['correctness'] == 'TRUE'].join(_.loc[_['correctness'] == 'FALSE'], rsuffix='_f', how='left')
    _['correctness_f'] = _['correctness_f'].fillna(value='FALSE')
    _['correctness'] = _['correctness'].fillna(value='TRUE')
    _[['index', 'index_f']] = _[['index', 'index_f']].fillna(value=0)
    _ = _[['index', 'index_f']].astype(int)
    _ = _.rename(index=str, columns={"index": "corrects", "index_f": "not_corrects"})
    return _
