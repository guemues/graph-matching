#!/usr/bin/env python

"""In this file i will create 100 * 10 noisy version of a main graph then compare the distances of the embedings."""
import pickle

__author__ = "Orcun Gumus"

from matching.simulation import Simulation
from matching.creation import RandomGraphType
from matching import EmbeddingType, ComparisonType

import time
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--run', dest='run', type=bool, help='')
parser.add_argument('--id', dest='id', type=str, help='')


parser.add_argument('--rgt', dest='random_graph_type', type=RandomGraphType, choices=list(RandomGraphType), help='')
parser.add_argument('--et', dest='embedding_type', type=EmbeddingType, choices=list(EmbeddingType), help='')
parser.add_argument('--ct', dest='comparison_type', type=ComparisonType, choices=list(ComparisonType), help='')

parser.add_argument('--d', dest='dimension_count', type=int, help='')
parser.add_argument('--e', dest='edge_probability', type=float, help='')
parser.add_argument('--n', dest='node_count', type=int, help='')

parser.add_argument('--s', dest='sample_size', type=int, help='')

parser.add_argument('--noise-step', dest='noise_step', type=float, help='')
parser.add_argument('--mn', dest='maximum_noise', type=float, help='')

parser.add_argument('--th-step', dest='th_step', type=float, help='')
parser.add_argument('--mt', dest='max_threshold', type=float, help='')

parser.add_argument('--ms', dest='main_sample', type=int, help='')
parser.add_argument('--hp', dest='hyperparamater', type=float, help='')


args = parser.parse_args()

test_id = str(int(time.time()))

if __name__ == '__main__':

    if args.id:
        with open('./results/{}'.format(args.id), 'rb') as f:
            simulation = pickle.load(f)
    else:
        simulation = Simulation(
            dimension_count=args.dimension_count,
            node_count=args.node_count,
            edge_probability=args.edge_probability,
            noise_step=args.noise_step,
            main_graph_sample_size=args.main_sample,
            sample_size=args.sample_size,
            maximum_noise=args.maximum_noise,
            embedding_type=args.embedding_type,
            th_step=args.th_step,
            max_threshold=args.max_threshold,
            graph_type=args.random_graph_type,
            hyperparameter=args.hyperparamater,
            test_id=test_id,
        )
    if args.run:
        simulation.run_nodes_mapping()

    # simulation.run_nodes_mapping()
    # simulation.run_correct_node_probability()
    # simulation.run_accuracy_tests()
    # simulation.run_distances_tests()
    simulation.save()

