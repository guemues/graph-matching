#!/usr/bin/env python

"""This module implemented for embedding matching."""
from collections import defaultdict, Counter
from enum import Enum
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import numpy as np

__all__ = ['compare_embeddings', 'map_embeddings_probabilities', 'match_using_threshold', 'confusion_matrix', 'ComparisonType']


class ComparisonType(Enum):
    Distribution = "DISTRIBUTION"
    Accuracy = "ACCURACY"


class MatchingType(Enum):
    Nearest = "nearest"
    Circle = "circle"


def distance_accuracy(distances):
    all_distance_mean = np.mean(distances)
    correct_distance_mean = np.mean(np.diag(distances))
    return max(0, 1 -correct_distance_mean / all_distance_mean)


def match_nearest(distances):

    assert distances.shape[0] == distances.shape[1]

    matches = {}

    matched_g1 = [False] * distances.shape[0]
    matched_g2 = [False] * distances.shape[0]

    sorted_indexes = np.dstack(np.unravel_index(np.argsort(distances.ravel()), (distances.shape[0], distances.shape[0]))).reshape(distances.shape[0] * distances.shape[0],2)
    for n1, n2 in sorted_indexes:
        if not matched_g1[n1] and not matched_g2[n2]:
            matches[n1] = n2

            matched_g1[n1] = True
            matched_g2[n2] = True

    assert all(matched_g1)
    assert all(matched_g2)

    return matches


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


def compare_difference_graph(graph_1, graph_2):
    """
    Check how many edges are different between two graph
    :param graph_1:
    :param graph_2:
    :return:
    """
    count = 0
    for edge in graph_1.edges():
        if edge not in graph_2.edges():
            count += 1

    for edge in graph_2.edges():
        if edge not in graph_1.edges():
            count += 1
    return count


def compare_embeddings(embedding_1, embedding_2, distance=euclidean):
    """

    :param embedding_1:
    :param embedding_2:
    :param distance:
    :return:
    """
    squares_sum = 0  # Iterative :(

    assert embedding_1.shape == embedding_2.shape

    for idx in range(embedding_1.shape[0]):
        squares_sum += distance(embedding_1[idx, :], embedding_2[idx, :])

    return squares_sum / embedding_1.shape[0]


def calculate_distances_iterative(embeddings_1, embeddings_2, distance=euclidean):
    """
    Calculate distances in between embeddings
    :param embeddings_1:
    :param embeddings_2:
    :param distance:
    :return:
    """
    assert embeddings_1.shape == embeddings_2.shape

    distances = np.zeros((embeddings_1.shape[0], embeddings_1.shape[0]))

    for idx_1 in range(embeddings_1.shape[0]):
        for idx_2 in range(embeddings_1.shape[0]):
            distances[idx_1, idx_2] = distance(embeddings_1[idx_1, :], embeddings_2[idx_2, :])

    return distances


def match_using_threshold(distances, ratio):
    """Match """

    neighbors = np.zeros(shape=distances.shape, dtype=np.uint8)
    neighbors[distances < ratio * np.max(distances)] = 1
    return neighbors

def confusion_matrix_one_to_one(matches, mapping_1, mapping_2):
    """

    :param mapping_2:
    :param mapping_1:
    :param matches:
    :return: tp, fp, fn, tn
    """
    tp = sum([1 for n1, n2 in matches.items() if n1 == n2])
    fp = sum([1 for n1, n2 in matches.items() if n1 != n2])
    fn = 0
    tn = len(matches) - fn - tp

    return tp, fp, fn, tn

def confusion_matrix(matches, mapping_1, mapping_2):
    """

    :param mapping_2:
    :param mapping_1:
    :param matches:
    :return: tp, fp, fn, tn
    """
    tp = np.sum(matches[list(mapping_1.values()), list(mapping_2.values())])
    fn = np.sum(matches[list(mapping_1.values()), list(mapping_2.values())] == 0)
    fp = np.sum(np.sum(matches) - tp)
    tn = np.sum(matches == 0) - fn - tp

    return tp, fp, fn, tn


def map_embeddings_probabilities(emb_1, emb_2, use_softmax=False):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    d_m = distance_matrix(emb_1, emb_2)
    p = d_m / np.sum(d_m, axis=1) if not use_softmax else np.apply_along_axis(softmax, 1, d_m)
    return p


if __name__ == '__main__':
    distances = np.array(
        [[5, 2, 4],
        [3, 3, 3],
        [6, 1, 2]]
    )
    print(match_nearest(distances))