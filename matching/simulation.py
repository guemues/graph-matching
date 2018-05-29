#!/usr/bin/env python
import csv

from networkx import degree_centrality, nx
from scipy.spatial import distance_matrix
from matching import create_main_graph, NoisyGraph, MainGraph
from matching.matching import MatchingType
from matching.statistics import find_degree_counts, find_counts, fill_empty, find_corrects_not_corrects, \
    find_node_counts
from .utils import mapping_dataframe, one_to_one_dataframe

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
            noise_step,
            sample_size,
            maximum_noise,
            embedding_type,
            graph_type,
            th_step,
            max_threshold,
            test_id,
            matching_type,
            verbose=True
    ):
        self.dimension_count = dimension_count
        self.node_count = node_count
        self.edge_probability = edge_probability
        self.noise_step = noise_step
        self.weights = [0.08, 1, 3.5]
        self.th_step = th_step

        self.sample_size = 2
        self.maximum_noise = maximum_noise

        self.noises = [0.03]
        self.thresholds = np.arange(0.01, 0.15, 0.01).tolist()

        self.matching_type = matching_type
        self.embedding_type = embedding_type
        self.graph_type = graph_type

        self.main_graphs = []
        self.noisy_graphs = []

        self.degrees = []
        self.degrees_count = []

        self.nodes_mapping = []
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
            'step': self.noise_step,
            'sample_size': self.sample_size,
            'maximum_noise': self.maximum_noise,
            'embedding_type': self.embedding_type.name,
            'graph_type': self.graph_type.name,
            'hyperparameter': self.weights
        }
        with open(RESULTS_JSON_FILENAME_FULL, 'w+') as fp:
            json.dump(results, fp)

        filename = os.path.join(RESULTS_FOLDER, str(self.test_id) + '.csv')

        def formatdata(data):
            for row in data:
                yield ["%0.2f" % v if isinstance(v, float) else str(v) for v in row ]

        with open(filename, 'w') as out:
            csv_out = csv.writer(out, delimiter=',')
            csv_out.writerow(['id', 'weight', 'noise', 'threshold', 'degree', 'node_count', 'tp', 'fp'])
            csv_out.writerows(formatdata(self.nodes_mapping))

        #
        # with open(RESULTS_JSON_FILENAME_FULL, 'w') as results_file:
        #     json.dump(results, results_file, indent=2)
        #
        # filename = os.path.join(RESULTS_FOLDER, self.test_id)
        #
        # nx.write_gpickle(self, filename)

    def _create_graphs(self):

        if len(self.main_graphs) > 0:
            print("No need for create graphs; they are already created.")
            return

        main_nx_graph = create_main_graph(
            graph_type=self.graph_type,
            node_count=self.node_count,
            edge_probability=self.edge_probability
        )
        main_graph = MainGraph(
            nx_graph=main_nx_graph,
            edge_probability=self.edge_probability,
            node_count=self.node_count,
        )
        self.main_graphs.append(main_graph)

        self.idx2degree = dict(main_nx_graph.degree)

        for idx, weight in enumerate(self.weights):
            weight_bin = []
            for jdx, edge_removal_probability in enumerate(self.noises):
                noisy_graphs_same_noise = [
                    NoisyGraph(
                        main_graph=main_graph,
                        edge_probability=self.edge_probability,
                        node_count=self.node_count,
                        edge_removal_probability=edge_removal_probability,
                        embedding_algorithm_enum=self.embedding_type,
                        dimension_count=self.dimension_count,
                        hyperparameter=weight
                    ) for _ in range(self.sample_size)]
                weight_bin.append(noisy_graphs_same_noise)

                if self.verbose:
                    print("{:0.2f}% of creation completed...".format((idx * len(self.noises) + jdx) / (len(self.noises) * len(self.weights)) * 100))

            self.noisy_graphs.append((weight, weight_bin))

    def _run(self, compare_function):

        self.main_graphs.clear()

        total_calculation = len(self.noises) * (self.sample_size * (self.sample_size - 1))
        current_calculation = 0

        graph_result = []

        for weight, weight_graph_bucket in self.noisy_graphs:
            for  noisy_graph_bucket  in weight_graph_bucket:
                for idx_1, noisy_graph in enumerate(noisy_graph_bucket):
                    for idx_2, compare_noisy_graph in enumerate(noisy_graph_bucket):

                        current_calculation += 1

                        assert isinstance(noisy_graph, NoisyGraph)
                        assert isinstance(compare_noisy_graph, NoisyGraph)

                        if idx_2 <= idx_1:
                            continue
                        distances = distance_matrix(noisy_graph.embeddings, compare_noisy_graph.embeddings, p=2)
                        small_result = compare_function(distances, self.thresholds, noisy_graph.mapping, compare_noisy_graph.mapping,compare_noisy_graph.noise, weight, self.idx2degree, self.test_id)

                        graph_result = graph_result + small_result

                        if self.verbose:
                            print('{:0.2f}% of run completed...'.format(int(current_calculation / total_calculation * 100)))
        return graph_result

    def run_nodes_mapping(self):
        if self.matching_type == MatchingType.Circle:
            func = mapping_dataframe

        elif self.matching_type == MatchingType.Nearest:
            func = one_to_one_dataframe

        self.nodes_mapping = self._run(func)


#
# def run_accuracy_tests(self):
#     self.accuracy_frame = self._run(accuracy_pd_row)
#
# def run_distances_tests(self):
#     self.distances_frame = self._run(distribution_pd_row)
