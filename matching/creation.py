#!/usr/bin/env python

"""This module implemented for graph and embedding creation."""

import numpy as np
import networkx as nx
import pandas as pd

from random import shuffle
from enum import Enum
from gem.embedding.gf import GraphFactorization
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.hope import HOPE

__all__ = ['MainGraph', 'NoisyGraph', 'EmbeddingType', 'RandomGraphType', 'create_main_graph']


def create_noisy_graph(nx_graph, noise, mixing=False):
    """
    This implementation is not in-place.

    :param mixing:
    :param nx_graph:
    :param q: Probability of changing the edge
    """
    nodes_count = len(nx_graph.nodes())

    if mixing:
        mapping = dict(zip(nx_graph.nodes(), np.random.permutation(nodes_count)))
    else:
        mapping = dict(zip(nx_graph.nodes(), nx_graph.nodes()))

    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    edges = list(nx_graph.edges())
    shuffle(edges)
    edges_selected = edges[0:int(len(nx_graph.edges) * (1 - noise))]

    graph = nx.Graph(nx_graph.edge_subgraph(edges_selected))
    mapping = mapping

    return graph, mapping


class EmbeddingType(Enum):
    LocallyLinearEmbedding = "LLE"
    Hope = "HOPE"
    GF = "GF"
    LaplacianEigenmaps = "LE"
    DegreeNeigDistributionWithout = "NEIGDEGREEWITHOUT"
    DegreeNeigDistribution = "NEIGDEGREE"
    DegreeNeigNeigDistribution = "NEIGNEIGDEGREE"


class RandomGraphType(Enum):
    GNP = "GNP"
    Powerlaw = "POWERLAW"


class Graph(object):

    def __init__(
            self
    ):
        self.graph = None
        pass

    def find_node_degrees(self):
        nodes, degrees = [], []
        for i, v in self.graph.degree:
            nodes.append(i)
            degrees.append(v)
        df = pd.DataFrame(
            data={
                "node": nodes,
                "degree": degrees
            })
        return df.set_index('node')


class MainGraph(Graph):

    def __init__(
            self,
            nx_graph,
            edge_probability,
            node_count
    ):

        super().__init__()
        self.edge_probability = edge_probability
        self.node_count = node_count
        self.graph = nx_graph
        self.max_degree = max(dict(nx_graph.degree).values())
        self.min_degree = min(dict(nx_graph.degree).values())
        self.degree_dist = list(dict(nx_graph.degree).values())


class NoisyGraph(Graph):

    def __init__(
            self,
            main_graph,
            edge_probability,
            node_count,
            edge_removal_probability,
            embedding_algorithm_enum,
            dimension_count,
            hyperparameter=1
    ):
        super().__init__()

        assert isinstance(embedding_algorithm_enum, EmbeddingType)

        self.edge_probability = edge_probability
        self.node_count = node_count

        self.main_graph = main_graph

        self.noise = edge_removal_probability

        self.mapping = None
        self.graph = None

        self.graph, self.mapping = create_noisy_graph(self.main_graph.graph, self.noise)

        self.e = get_embeddings(self.graph, embedding_algorithm_enum, dimension_count, hyperparameter, self.main_graph.min_degree, self.main_graph.max_degree)

        del self.graph
        self.graph = None

    @property
    def embeddings(self):
        return self.e


def get_embeddings(graph, embedding_algorithm_enum, dimension_count, hyperparameter, lower=None, higher=None):
    """Generate embeddings. """

    if embedding_algorithm_enum is EmbeddingType.LocallyLinearEmbedding:
        embedding_alg = LocallyLinearEmbedding(d=dimension_count)
    elif embedding_algorithm_enum is EmbeddingType.Hope:
        embedding_alg = HOPE(d=dimension_count, beta=0.01)
    elif embedding_algorithm_enum is EmbeddingType.GF:
        embedding_alg = GraphFactorization(d=dimension_count, max_iter=100000, eta=1 * 10 ** -4, regu=1.0)
    elif embedding_algorithm_enum is EmbeddingType.LaplacianEigenmaps:
        embedding_alg = LaplacianEigenmaps(d=dimension_count)

    elif embedding_algorithm_enum is EmbeddingType.DegreeNeigDistributionWithout:
        A = np.array([np.histogram([graph.degree(neig) for neig in graph.neighbors(i)], bins=dimension_count, density=True, range=(lower, higher))[0] for i in graph.nodes()])
        return A

    elif embedding_algorithm_enum is EmbeddingType.DegreeNeigDistribution:
        A = np.array([np.concatenate([np.array([graph.degree(i) / (higher * dimension_count)]) , np.histogram([graph.degree(neig) for neig in graph.neighbors(i)], bins=dimension_count - 1, density=True, range=(lower, higher))[0]], axis=0) for i in graph.nodes()])
        return A

    elif embedding_algorithm_enum is EmbeddingType.DegreeNeigNeigDistribution:
        A = np.array( [
            np.concatenate([np.array([
                (hyperparameter) * graph.degree(i) / (higher * dimension_count)]) ,
                (hyperparameter * hyperparameter) * np.histogram([graph.degree(neig) for neig in graph.neighbors(i)], bins=int(dimension_count / 2), density=True, range=(lower, higher))[0] ,
                (hyperparameter * hyperparameter * hyperparameter) * np.histogram([graph.degree(neigneig) for neig in graph.neighbors(i) for neigneig in graph.neighbors(neig)], bins=int(dimension_count / 2), density=True, range=(lower, higher))[0]], axis=0)
            for i in graph.nodes()]
        )

        return A
    else:
        raise NotImplementedError

    e, t = embedding_alg.learn_embedding(graph=graph, no_python=True)

    e = np.dot(e, np.diag(np.sign(np.mean(e, axis=0))))
    return e


def create_main_graph(graph_type, **kwargs):

    assert isinstance(graph_type, RandomGraphType)

    if graph_type is RandomGraphType.GNP:
        assert 'edge_probability' in kwargs
        assert 'node_count' in kwargs

        edge_probability = kwargs['edge_probability']
        node_count = kwargs['node_count']

        return nx.gnp_random_graph(node_count, edge_probability)

    if graph_type is RandomGraphType.Powerlaw:
        assert 'edge_probability' in kwargs
        assert 'node_count' in kwargs

        edge_probability = kwargs['edge_probability']
        node_count = kwargs['node_count']

        edge_count = int(edge_probability * node_count)
        return nx.powerlaw_cluster_graph(node_count, edge_count, .0)
