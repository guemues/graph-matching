import numpy as np
import itertools
import networkx as nx

__all__ = ['noisy_version']


def noisy_version(graph: nx.Graph, q=0.01) -> nx.Graph:
    """
    Not in-place
    :param q: Probability of changing the edge
    """

    def toggle_edge(graph: nx.Graph, source, target) -> nx.Graph:
        """
        In-place
        If there is a edge in-between source and target remove it. If there is not, add it.

        :param source: Edge source
        :param target: Edge target
        """
        if graph.has_edge(source, target):
            graph.remove_edge(source, target)
        else:
            graph.add_edge(source, target)

        return graph

    copy_graph = graph.copy()
    all_possible_node_pairs = [i for i in itertools.product(copy_graph.nodes(), copy_graph.nodes())]

    which_possible_node_pairs_will_effect = np.random.binomial(1, q, len(all_possible_node_pairs))

    for idx, (source_node, target_node) in enumerate(all_possible_node_pairs):
        if which_possible_node_pairs_will_effect[idx] == 1:
            toggle_edge(copy_graph, source_node, target_node)

    return copy_graph


