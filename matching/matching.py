import numpy as np
import logging

from scipy.spatial.distance import euclidean
from sympy.utilities.iterables import multiset_permutations

__all__ = ['compare_ordered_embeddings', 'best_compare_swapping_embeddings', 'best_compare_mirror_embeddings', 'compare_difference_graph']

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


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


def swap_dimensions(embedding, dimension_1, dimension_2):
    """
       Swap row and column i and j in-place.

       Examples
       --------
       >>> cm = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
       >>> swap_dimensions(cm, 2, 0)
       array([[2, 1, 0],
              [5, 4, 3],
              [8, 7, 6]])
       """

    temp = np.copy(embedding[:, dimension_1])
    embedding[:, dimension_1] = embedding[:, dimension_2]
    embedding[:, dimension_2] = temp
    return embedding


def best_compare_mirror_embeddings(embedding_1, embedding_2, distance=euclidean):
    """
    Try to find best embedding dimensions by changing sign of the dimensions

    :param embedding_1:
    :param embedding_2:
    """
    dimension_len = embedding_1.shape[1]

    def find_all_sign_possibilities(*arrays):
        grid = np.meshgrid(*arrays)
        coord_list = [entry.flatten() for entry in grid]
        return np.vstack(coord_list).T.tolist()

    all_sign_possibilities = find_all_sign_possibilities(*(dimension_len * [np.array([-1, 1])]))

    points = [compare_ordered_embeddings(embedding_1, np.dot(embedding_2, np.diag(np.array(sign_p))), distance=distance) for sign_p in all_sign_possibilities]

    logger.info(points)

    return min(points)


def best_compare_swapping_embeddings(embedding_1, embedding_2):
    """
    Try to find best embedding dimensions by changing order of the dimensions

    :param embedding_1:
    :param embedding_2:
    """
    assert embedding_1.shape == embedding_2.shape

    dimensions_count = embedding_1.shape[1]
    all_permutations = multiset_permutations(np.arange(dimensions_count))
    points = [compare_ordered_embeddings(embedding_1, embedding_2[:, i]) for i in all_permutations]

    logger.info(points)

    return min(points)


def compare_ordered_embeddings(embedding_1, embedding_2, distance=euclidean):
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


def calculate_distances(embeddings_1, embeddings_2, distance=euclidean):
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

# def mapping_embeddings(embeddings_1, embeddings_2, distance=euclidean):
#
#
#     return distances