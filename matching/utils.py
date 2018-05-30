"""This module implemented for visualizations."""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from matching import match_using_threshold, confusion_matrix
from matching.matching import confusion_matrix_one_to_one, match_nearest


def one_to_one_matching(distances, thresholds, mapping_1, mapping_2,  noise, hyperparameter, degrees, test_id):
    matches = match_nearest(distances)
    tp, fp, fn, tn = confusion_matrix_one_to_one(matches, mapping_1, mapping_2)

    return [(test_id, hyperparameter, noise, tp, fp)]



def one_to_many_matching(distances, thresholds,  mapping_1, mapping_2, noise, hyperparameter, idx2degree, test_id):
    """

    :param distances: For every node i and j in main graph G distance matrix
    contains (i, j) entries which gives the distance of the embedding of G' for i and
    j from G''. G' and G'' are the noisy versions of the graph G.

    :param mapping_1: For every node i in the main graph G and associated node i'
    which is in the noisy graph G', mapping_1 contains i': i

    :param mapping_2: For every node j in the main graph G and associated node i'
    which is in the noisy graph G'', mapping_2 contains j': j

    :param threshold_ratio: TO determine mathes in between two noisy graphs G' and G''
    threshold value determined via max(distance) * threshold_ratio)

    :return: For every node i in graph G it returns a pandas dataframe row with
    every node j which are close enough (distance < threshold_ratio * max(distance))
    in the embeddings of G'(i') and G''(j')
    """
    results = []

    degree2count = defaultdict(int)
    for node_id, degree in idx2degree.items():
        degree2count[degree] += 1

    for threshold_ratio in thresholds:
        match = match_using_threshold(distances, threshold_ratio)
        i_noisy, j_noisy = np.where(match == 1)

        trues_degrees = []; false_degrees = []
        for idx, i in enumerate(i_noisy):
            if mapping_1[i_noisy[idx]] == mapping_2[j_noisy[idx]]:
                trues_degrees.append(idx2degree[mapping_1[i_noisy[idx]]])
            else:
                false_degrees.append(idx2degree[mapping_2[j_noisy[idx]]])

        true_counter = Counter(trues_degrees)
        false_counter = Counter(false_degrees)

        results += [(test_id, hyperparameter, noise, threshold_ratio, degree, degree2count[degree], true_counter[degree], false_counter[degree]) for degree in degree2count.keys()]


    return sorted(results, reverse=True, key=lambda tup: (tup[1],tup[2],tup[3],tup[4]) )

    #
    # tp_mask = matches[list(mapping_1.values()), list(mapping_2.values())] == 1
    #
    # tps, tns = correctly_estimated_nodes(match, mapping_1, mapping_2)
    #
    # _ = pd.DataFrame(
    #     data={
    #         "node": tps,
    #         "correctness": pd.Series(["CORRECT"] * len(tps)),
    #         "threshold_ratio": pd.Series([threshold] * len(tps))
    #     }
    # )
    # _2 = pd.DataFrame(
    #     data={
    #         "node": tns,
    #         "correctness": pd.Series(["FALSE"] * len(tns)),
    #         "threshold_ratio": pd.Series([threshold] * len(tns))
    #     }
    # )
    #
    # df = pd.concat([df, _, _2])



def distribution_pd_row(distances, mapping_1, mapping_2, edge_removal_possibility):

    correct_distances = distances[list(mapping_1.values()), list(mapping_2.values())]
    correct_mapping_mask = np.ones(distances.shape, dtype=bool)
    correct_mapping_mask[list(mapping_1.values()), list(mapping_2.values())] = False
    incorrect_distances = np.random.choice(distances[correct_mapping_mask], 1000)

    correctness_series = pd.Series(["CORRECT"] * correct_distances.shape[0] + ["FALSE"] * incorrect_distances.shape[0], dtype="category")
    df = pd.DataFrame(
        data={
            "edge_removal_possibility": edge_removal_possibility,
            "distances": np.concatenate((correct_distances, incorrect_distances)),
            "correctness": correctness_series
        }
    )
    return df


# def plot_distributions_from_pandas(df_all):
#
#     for dimension_count in df_all.dimensions.unique():
#         for algorithm in df_all.algorithm.unique():
#             fig, ax = plt.subplots(nrows=2, ncols=len(df_all.q_value.unique()), sharex='col', sharey='row', figsize=(20, 8))
#
#             for idx, q_value in enumerate(df_all.q_value.unique()):
#
#                 df = df_all[df_all["algorithm"] == algorithm]
#                 df = df[df["q_value"] == q_value]
#                 df = df[df["dimensions"] == dimension_count]
#
#                 fig.suptitle("Alg: {}, Dim: {}".format(algorithm, dimension_count), fontsize=10)
#                 ax[0, idx].set_title("Q: {}".format(q_value), fontsize=10)
#
#                 g1 = sns.kdeplot(df[df["correctness"] == "FALSE"].distances, ax=ax[0, idx], shade=True, cumulative=False, label='False')
#                 g2 = sns.kdeplot(df[df["correctness"] == "CORRECT"].distances, ax=ax[0, idx], shade=True, cumulative=False, label='Correct')
#
#                 g1.set(ylim=(0, 500))
#                 g2.set(ylim=(0, 500))
#
#                 g1_cum = sns.kdeplot(df[df["correctness"] == "FALSE"].distances, ax=ax[1, idx], shade=True, cumulative=True,
#                                  label='False')
#                 g2_cum = sns.kdeplot(df[df["correctness"] == "CORRECT"].distances, ax=ax[1, idx], shade=True, cumulative=True,
#                                  label='Correct')
#
#                 g1_cum.set(ylim=(0, 1.15))
#                 g2_cum.set(ylim=(0, 1.15))
#
#             plt.show()
