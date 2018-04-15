#!/usr/bin/env python
from networkx import degree_centrality, nx
from scipy.spatial import distance_matrix
from matching import create_main_graph, NoisyGraph, MainGraph
from matching.statistics import find_degree_counts, find_counts, fill_empty, find_corrects_not_corrects, \
    find_node_counts
from .utils import mapping_dataframe

import json
import os
import numpy as np
import pandas as pd
import collections

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
            main_graph_sample_size,
            sample_size,
            maximum_noise,
            embedding_type,
            graph_type,
            max_threshold,
            test_id,
            verbose=True
    ):
        self.dimension_count = dimension_count
        self.node_count = node_count
        self.edge_probability = edge_probability
        self.step = step
        self.sample_size = sample_size
        self.maximum_noise = maximum_noise

        self.noises = list(np.arange(0, self.maximum_noise, self.step))
        self.thresholds = np.arange(0.01, max_threshold, 0.01)

        self.embedding_type = embedding_type
        self.graph_type = graph_type
        self.main_graph_sample_size = main_graph_sample_size

        self.main_graphs = []
        self.noisy_graphs = []

        self.degrees = []
        self.degrees_count = []

        self.nodes_mapping = pd.DataFrame()
        self.max_threshold = max_threshold
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

        if len(self.main_graphs) > 0:
            print("No need for create graphs; they are already created.")
            return

        for _ in range(self.main_graph_sample_size):
            main_nx_graph = create_main_graph(
                graph_type=self.graph_type,
                node_count=self.node_count,
                edge_probability=self.edge_probability
            )
            main_graph = MainGraph(
                nx_graph=main_nx_graph,
                edge_probability=self.edge_probability,
                node_count=self.node_count,
                embedding_algorithm_enum=self.embedding_type,
                dimension_count=self.dimension_count
            )
            self.main_graphs.append(main_graph)

            self.degrees.append(dict(main_nx_graph.degree))
            self.degrees_count.append(dict(collections.Counter(sorted([d for n, d in main_nx_graph.degree()], reverse=True))))
            # for main_nx_graph.degree
            # self.degree_c = degree_centrality(main_nx_graph)

            self.noisy_graphs.append([[
                    NoisyGraph(
                        main_graph=main_graph,
                        edge_probability=self.edge_probability,
                        node_count=self.node_count,
                        edge_removal_probability=edge_removal_probability,
                        embedding_algorithm_enum=self.embedding_type,
                        dimension_count=self.dimension_count
                    ) for _ in range(self.sample_size)] for edge_removal_probability in self.noises])
            print()
            print()

    def _run(self, compare_function):

        self.main_graphs.clear()

        total_calculation = self.main_graph_sample_size * len(self.noises) * self.sample_size * self.sample_size
        current_calculation = 0
        result = pd.DataFrame()

        for main_graph_idx, noisy_graph_samples in enumerate(self.noisy_graphs):
            graph_result = pd.DataFrame()

            for noisy_graph_bucket in noisy_graph_samples:
                for idx_1, noisy_graph in enumerate(noisy_graph_bucket):
                    for idx_2, compare_noisy_graph in enumerate(noisy_graph_bucket):

                        current_calculation += 1

                        assert isinstance(noisy_graph, NoisyGraph)
                        assert isinstance(compare_noisy_graph, NoisyGraph)

                        if idx_2 <= idx_1:
                            continue
                        distances = distance_matrix(noisy_graph.embeddings, compare_noisy_graph.embeddings, p=2)
                        small_result = compare_function(distances, self.thresholds, noisy_graph.mapping, compare_noisy_graph.mapping,compare_noisy_graph.noise, self.degrees[main_graph_idx], main_graph_idx)
                        # Stats
                        graph_result = pd.concat([graph_result, small_result])

                        if self.verbose:
                            print('%{} completed for run'.format(int(current_calculation / total_calculation * 100)),  end="\r", flush=True)

            degree_count = self.degrees_count[main_graph_idx]
            degree_counts = find_degree_counts(degree_count)
            mapping_df_find_counts = find_counts(graph_result)
            mapping_df_fill_empty = fill_empty(mapping_df_find_counts, self.thresholds, self.noises,
                                               list(degree_count.keys()))
            mapping_df_find_corrects_not_corrects = find_corrects_not_corrects(mapping_df_fill_empty)
            mapping_df_find_node_counts = find_node_counts(mapping_df_find_corrects_not_corrects, degree_counts, int((len(noisy_graph_bucket) * (len(noisy_graph_bucket) - 1)) / 2) )
            mapping_df_find_node_counts['main_graph'] = main_graph_idx
            result = pd.concat([result, mapping_df_find_node_counts])

        return result

    def run_nodes_mapping(self):
        self.nodes_mapping  = self._run(mapping_dataframe)

#
# def run_accuracy_tests(self):
#     self.accuracy_frame = self._run(accuracy_pd_row)
#
# def run_distances_tests(self):
#     self.distances_frame = self._run(distribution_pd_row)
