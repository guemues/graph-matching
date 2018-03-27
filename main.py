#!/usr/bin/env python

"""In this file i will create 100 * 10 noisy version of a main graph then compare the distances of the embedings."""

__author__ = "Orcun Gumus"

from matching.simulation import Simulation
from matching.creation import RandomGraphType
from matching import EmbeddingType, ComparisonType

import time
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--rgt', dest='random_graph_type', type=RandomGraphType, choices=list(RandomGraphType), help='')
parser.add_argument('--et', dest='embedding_type', type=EmbeddingType, choices=list(EmbeddingType), help='')
parser.add_argument('--ct', dest='comparison_type', type=ComparisonType, choices=list(ComparisonType), help='')

parser.add_argument('--d', dest='dimension_count', type=int, help='')
parser.add_argument('--e', dest='edge_probability', type=float, help='')
parser.add_argument('--n', dest='node_count', type=int, help='')
parser.add_argument('--step', dest='step', type=float, help='')
parser.add_argument('--s', dest='sample_size', type=int, help='')
parser.add_argument('--mn', dest='maximum_noise', type=float, help='')


args = parser.parse_args()

test_id = str(int(time.time()))

file_name = './results/{}.pickle'.format(test_id)

if __name__ == '__main__':

    simulation = Simulation(
        dimension_count=args.dimension_count,
        node_count=args.node_count,
        edge_probability=args.edge_probability,
        step=args.step,
        sample_size=args.sample_size,
        maximum_noise=args.maximum_noise,
        embedding_type=args.embedding_type,
        graph_type=args.random_graph_type,
        test_id=test_id
    )
    # simulation.run_nodes_mapping()
    # simulation.run_correct_node_probability()
    # simulation.run_accuracy_tests()
    # simulation.run_distances_tests()
    simulation.save()

