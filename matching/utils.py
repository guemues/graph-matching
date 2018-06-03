"""This module implemented for visualizations."""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from matching import match_using_threshold, confusion_matrix
from matching.matching import confusion_matrix_one_to_one, match_nearest



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
