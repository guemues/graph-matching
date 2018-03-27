#!/usr/bin/env python
from networkx import degree_centrality, nx
from scipy.spatial import distance_matrix
from matching import create_main_graph, NoisyGraph, MainGraph
from .utils import mapping_dataframe

import json
import os
import numpy as np
import pandas as pd
import pickle

RESULTS_FOLDER = './results'
RESULTS_JSON_FILENAME = 'info.json'
RESULTS_JSON_FILENAME_FULL = os.path.join(RESULTS_FOLDER, RESULTS_JSON_FILENAME)

class Simulation(object):

    def __init__(
            self,
            dimension_count,
            node_count,
            edge_probability,
            step,
            sample_size,
            maximum_noise,
            embedding_type,
            graph_type,
            test_id,
            verbose=True
    ):
        self.dimension_count = dimension_count
        self.node_count = node_count
        self.edge_probability = edge_probability
        self.step = step
        self.sample_size = sample_size
        self.maximum_noise = maximum_noise
        self.embedding_type = embedding_type
        self.graph_type = graph_type

        self.main_graph = None
        self.noisy_graphs = None

        self.degree = None
        self.degree_c = None

        self.nodes_mapping = pd.DataFrame()

        self.test_id = test_id

        self.verbose = verbose
        self._create_graphs()

    def load(self):
        folder = os.path.join(RESULTS_FOLDER, self.test_id)
        self.main_graph = nx.read_gpickle(os.path.join(folder, 'nx_main_graph'))

    def save(self):
        if not os.path.exists(RESULTS_FOLDER):
            os.makedirs(RESULTS_FOLDER)

        if os.path.isfile(RESULTS_JSON_FILENAME_FULL):
            with open(RESULTS_JSON_FILENAME_FULL, 'r') as results_file:
                results = json.load(results_file)
        else:
            results = {}

        results[self.test_id] = {
            'dimension_count': self.dimension_count,
            'node_count': self.node_count,
            'edge_probability': self.edge_probability,
            'step': self.step,
            'sample_size': self.sample_size,
            'maximum_noise': self.maximum_noise,
            'embedding_type': self.embedding_type.name,
            'graph_type': self.graph_type.name
        }

        with open(RESULTS_JSON_FILENAME_FULL, 'w') as results_file:
            json.dump(results, results_file, indent=2)

        filename = os.path.join(RESULTS_FOLDER, self.test_id)

        nx.write_gpickle(self, filename)

    def _create_graphs(self):

        if self.main_graph:
            print("No need for create graphs; they are already created.")
            return

        main_nx_graph = create_main_graph(
            graph_type=self.graph_type,
            node_count=self.node_count,
            edge_probability=self.edge_probability
        )

        self.main_graph = MainGraph(
            nx_graph=main_nx_graph,
            edge_probability=self.edge_probability,
            node_count=self.node_count,
            embedding_algorithm_enum=self.embedding_type,
            dimension_count=self.dimension_count
        )

        self.degree = main_nx_graph.degree
        self.degree_c = degree_centrality(main_nx_graph)

        self.noisy_graphs = [
            [
                NoisyGraph(
                    main_graph=self.main_graph,
                    edge_probability=self.edge_probability,
                    node_count=self.node_count,
                    edge_removal_probability=edge_removal_probability,
                    embedding_algorithm_enum=self.embedding_type,
                    dimension_count=self.dimension_count
                ) for _ in range(self.sample_size)] for edge_removal_probability in
            np.arange(0, self.maximum_noise, self.step)]

    def _run(self, compare_function):
        total_calculation = len(self.noisy_graphs) * self.sample_size * self.sample_size
        current_calculation = 0
        result = pd.DataFrame()
        for noisy_graph_bucket in self.noisy_graphs:

            for idx_1, noisy_graph in enumerate(noisy_graph_bucket):
                for idx_2, compare_noisy_graph in enumerate(noisy_graph_bucket):

                    current_calculation += 1

                    assert isinstance(noisy_graph, NoisyGraph)
                    assert isinstance(compare_noisy_graph, NoisyGraph)

                    if idx_2 <= idx_1:
                        continue
                    distances = distance_matrix(noisy_graph.embeddings, compare_noisy_graph.embeddings, p=2)
                    results_current = compare_function(distances, noisy_graph.mapping, compare_noisy_graph.mapping,compare_noisy_graph.noise)

                    result = pd.concat([result, results_current])

                    if self.verbose:
                        print('%{} completed for run'.format(int(current_calculation / total_calculation * 100)))
        return result

    def run_nodes_mapping(self):
        self.nodes_mapping = self._run(mapping_dataframe)

#
# def run_accuracy_tests(self):
#     self.accuracy_frame = self._run(accuracy_pd_row)
#
# def run_distances_tests(self):
#     self.distances_frame = self._run(distribution_pd_row)
