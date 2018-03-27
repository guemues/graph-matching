"""This module implemented for visualizations."""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matching import match_using_threshold, confusion_matrix


def mapping_dataframe(distances,  mapping_1, mapping_2, noise):
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

    df = pd.DataFrame()
    for threshold_ratio in np.arange(0, 0.05, 0.005):
        match = match_using_threshold(distances, threshold_ratio)
        i_noisy, j_noisy = np.where(match == 1)

        if len(i_noisy) == 0 and len(j_noisy) == 0:
            continue

        df = pd.concat([df, pd.DataFrame(
            data={
                "threshold_ratio": threshold_ratio,
                "noise": noise,
                "node_1": np.vectorize(mapping_1.get)(i_noisy),
                "node_2": np.vectorize(mapping_2.get)(j_noisy)
            })
        ])
    return df.reset_index()

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


def correct_node_probability(distances, mapping_1, mapping_2):
    df = pd.DataFrame()
    for t in np.arange(0, 0.1, 0.002):
        match = match_using_threshold(distances, t)
        tps, fps = confusion_matrix(match, mapping_1, mapping_2)

        _ = pd.DataFrame(
                data={
                    "node": tps,
                    "correctness": pd.Series(["CORRECT"] * len(tps)),
                    "threshold_ratio": pd.Series([t] * len(tps))
                }
            )
        _2 = pd.DataFrame(
            data={
                "node": fps,
                "correctness": pd.Series(["FALSE"] * len(fps)),
                "threshold_ratio": pd.Series([t] * len(fps))
                }
            )

        df = pd.concat([df, _, _2])

    return df


def tp_fp_fn_tn_pd_row(distances, mapping_1, mapping_2):

    tps, fps, fns, tns, ts = [], [], [], [], []
    for t in np.arange(0, 0.5, 0.002):
        match = match_using_threshold(distances, t)
        tp, fp, fn, tn = confusion_matrix(match, mapping_1, mapping_2)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        ts.append(t)

    df = pd.DataFrame(
        data={
            "true_positive": tps,
            "false_positive": fps,
            "false_negative": fns,
            "true_negative": tns,
            "threshold_ratio": ts,
        }
    )

    return df


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


def plot_distributions_from_pandas(df_all):

    for dimension_count in df_all.dimensions.unique():
        for algorithm in df_all.algorithm.unique():
            fig, ax = plt.subplots(nrows=2, ncols=len(df_all.q_value.unique()), sharex='col', sharey='row', figsize=(20, 8))

            for idx, q_value in enumerate(df_all.q_value.unique()):

                df = df_all[df_all["algorithm"] == algorithm]
                df = df[df["q_value"] == q_value]
                df = df[df["dimensions"] == dimension_count]

                fig.suptitle("Alg: {}, Dim: {}".format(algorithm, dimension_count), fontsize=10)
                ax[0, idx].set_title("Q: {}".format(q_value), fontsize=10)

                g1 = sns.kdeplot(df[df["correctness"] == "FALSE"].distances, ax=ax[0, idx], shade=True, cumulative=False, label='False')
                g2 = sns.kdeplot(df[df["correctness"] == "CORRECT"].distances, ax=ax[0, idx], shade=True, cumulative=False, label='Correct')

                g1.set(ylim=(0, 500))
                g2.set(ylim=(0, 500))

                g1_cum = sns.kdeplot(df[df["correctness"] == "FALSE"].distances, ax=ax[1, idx], shade=True, cumulative=True,
                                 label='False')
                g2_cum = sns.kdeplot(df[df["correctness"] == "CORRECT"].distances, ax=ax[1, idx], shade=True, cumulative=True,
                                 label='Correct')

                g1_cum.set(ylim=(0, 1.15))
                g2_cum.set(ylim=(0, 1.15))

            plt.show()
