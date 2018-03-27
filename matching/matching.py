#!/usr/bin/env python

"""This module implemented for embedding matching."""


from enum import Enum, auto
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import numpy as np

__all__ = ['compare_embeddings', 'map_embeddings_probabilities', 'match_using_threshold', 'confusion_matrix', 'ComparisonType']


class ComparisonType(Enum):
    Distribution = "DISTRIBUTION"
    Accuracy = "ACCURACY"


class MatchingType(Enum):
    Nearest = auto()
    Circle = auto()


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


def correctly_estimated_nodes(matches, mapping_1, mapping_2):
    """
    # TODO: ***
    :param mapping_2:
    :param mapping_1:
    :param matches:
    :return: tp, fp, fn, tn
    """

    tp_mask = matches[list(mapping_1.values()), list(mapping_2.values())] == 1
    fp_mask = matches[list(mapping_1.values()), list(mapping_2.values())] == 0
    fn_mask = matches[ones]

    tp = np.array(list(mapping_1.values()))[tp_mask].tolist()
    fp = np.array(list(mapping_1.values()))[fp_mask].tolist()

    return tp, tn


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
