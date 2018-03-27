#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns


def confusion(simulation, node_degrees, degree_counts, threshold_ratio=0.040, noise=0.01, min_node_degree=0):
    new_frame = simulation.nodes_mapping[simulation.nodes_mapping['threshold_ratio'] == threshold_ratio]
    new_frame = new_frame[new_frame['noise'] == noise]

    new_frame = new_frame.join(node_degrees, on='node_1', rsuffix='_1')
    new_frame = new_frame.join(node_degrees, on='node_2', rsuffix='_2')

    new_frame = new_frame[new_frame['degree'] > min_node_degree]
    new_frame = new_frame[new_frame['degree_2'] > min_node_degree]

    new_frame['correctness'] = (new_frame['node_1'] == new_frame['node_2']).astype(str)
    # new_frame[new_frame['node_1'] == new_frame['node_2']]['correcness'] = 0

    new_frame.groupby(['degree', 'correctness'])['index'].count()

    new_frame = new_frame.pivot_table(
        index=['degree', 'correctness'],
        values='index',
        fill_value=0,
        aggfunc='count').unstack()

    new_frame = new_frame.fillna(0)
    # new_frame.reset_index()
    try:
        new_frame.columns = ['false_positive', 'true_positive']
    except:
        print(len(new_frame))
        return
    new_frame = new_frame.join(degree_counts)

    new_frame['false_negative'] = new_frame['node'] - new_frame['true_positive']
    new_frame['true_negative'] = (new_frame['node'] * 999) - new_frame['false_positive']

    # new_frame['total'] = new_frame['False'] + new_frame['True']

    new_frame['greater_false_positive'] = new_frame.loc[::-1, 'false_positive'].cumsum()[::-1]
    new_frame['greater_true_positive'] = new_frame.loc[::-1, 'true_positive'].cumsum()[::-1]
    new_frame['greater_false_negative'] = new_frame.loc[::-1, 'false_negative'].cumsum()[::-1]
    new_frame['greater_true_negative'] = new_frame.loc[::-1, 'true_negative'].cumsum()[::-1]

    new_frame['precision'] = new_frame['true_positive'] / (new_frame['true_positive'] + new_frame['false_negative'])
    new_frame['sensivity'] = new_frame['true_positive'] / (new_frame['true_positive'] + new_frame['false_positive'])
    new_frame['accuracy'] = (new_frame['true_positive'] + new_frame['true_negative']) / (
            new_frame['true_positive'] + new_frame['true_negative'] + new_frame['false_positive'] + new_frame[
        'false_negative'])

    new_frame['greater_precision'] = new_frame['greater_true_positive'] / (
            new_frame['greater_true_positive'] + new_frame['greater_false_negative'])
    new_frame['greater_sensivity'] = new_frame['greater_true_positive'] / (
            new_frame['greater_true_positive'] + new_frame['greater_false_positive'])
    new_frame['greater_accuracy'] = (new_frame['greater_true_positive'] + new_frame['greater_true_negative']) / (
            new_frame['greater_true_positive'] + new_frame['greater_true_negative'] + new_frame[
        'greater_false_positive'] + new_frame['greater_false_negative'])

    new_frame = new_frame.reset_index()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    ax1.set_title('Precision Sensivity Graph')
    ax1.set(xlabel='Node Degree', ylabel='Precision / Sensivity')

    ax1.plot('degree', 'greater_precision', data=new_frame, marker='o', label='Precision')
    ax1.plot('degree', 'greater_sensivity', data=new_frame, marker='o', label='Sensivity')

    ax2.plot('degree', 'greater_accuracy', data=new_frame, color='gray', marker='o', label='Accuracy')
    ax2.set_title('Accuracy Graph')
    ax2.set(xlabel='Node Degree', ylabel='Accuracy')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)


def average_matching_noise(simulation):
    a = simulation.nodes_mapping.groupby(['noise']).count()['index'].reset_index()
    a['index'] = a['index'] / (1000 * len(simulation.nodes_mapping['threshold_ratio'].unique()))
    ax = sns.lmplot(x="noise", y="index", data=a)
    ax.set(xlabel='Noise', ylabel='Average Match for one Node')
    # plt.plot(a['index'].count().index, a['index'].count().as_matrix(), 'ro')


def average_matching_threshold(simulation):
    a = simulation.nodes_mapping.groupby(['threshold_ratio']).count()['index'].reset_index()
    a['index'] = a['index'] / (1000 * len(simulation.nodes_mapping['noise'].unique()))
    ax = sns.lmplot(x="threshold_ratio", y="index", data=a, lowess=True)
    ax.set(xlabel='Threshold', ylabel='Average Match for one Node')


def degree_distribution(simulation, field='degree'):
    nodes_frame = simulation.nodes_frame

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Degree Distribution of Matches')
    ax.set(xlabel='Node Degree', ylabel='Probability')

    corrects = nodes_frame[nodes_frame['correctness'] == 'CORRECT']
    falses = nodes_frame[nodes_frame['correctness'] == 'FALSE']

    sns.kdeplot(corrects[field], ax=ax, shade=True, color="g", label="Correct")
    sns.kdeplot(falses[field], ax=ax, shade=True, color="r", label="Wrong")
    plt.show()

    print('Correct matches statistics')
    print(corrects[field].describe())
    print('\nWrong matches statistics')
    print(falses[field].describe())


def accuracy_and_power_for_noise(simulation, edge_removal_possibility):
    accuracy_frame = simulation.accuracy_frame

    accuracy_frame = accuracy_frame[accuracy_frame['edge_removal_possibility'] == edge_removal_possibility].groupby('threshold_ratio').mean()

    accuracy_frame['accuracy'] = accuracy_frame['true_positive'] / (accuracy_frame['true_positive'] + accuracy_frame['false_positive'])
    accuracy_frame['power'] = accuracy_frame['true_positive'] / 1000

    accuracy_frame = accuracy_frame[accuracy_frame.accuracy > 0.1]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Accuracy and Power of the Matching')
    ax.set(xlabel='Threshold Ratio', ylabel='Proportion')

    ax.plot(accuracy_frame.index, accuracy_frame['power'], color="b", label="Power")
    ax.plot(accuracy_frame.index, accuracy_frame['accuracy'], color="g", label="Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def accuracy_and_power_for_threshold(simulation, threshold_ratio):
    accuracy_frame = simulation.accuracy_frame

    accuracy_frame = accuracy_frame[accuracy_frame['threshold_ratio'] == threshold_ratio].groupby('edge_removal_possibility').mean()

    accuracy_frame['accuracy'] = accuracy_frame['true_positive'] / (accuracy_frame['true_positive'] + accuracy_frame['false_positive'])
    accuracy_frame['power'] = accuracy_frame['true_positive'] / 1000

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Accuracy and Power of the Matching')
    ax.set(xlabel='Noise', ylabel='Proportion')

    ax.plot(accuracy_frame.index, accuracy_frame['power'], color="b", label="Power")
    ax.plot(accuracy_frame.index, accuracy_frame['accuracy'], color="g", label="Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def distribution_graph(simulation, edge_removal_possibility):
    data = simulation.distances_frame

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(221)
    ax.set_title('Distance Distributions')
    ax.set(xlabel='Euclidean Distance', ylabel='Probability')
    selected_data = data[data["edge_removal_possibility"] == edge_removal_possibility]
    g1 = sns.kdeplot(selected_data[selected_data["correctness"] == "FALSE"].distances, ax=ax,
                     color=(0.86, 0.5712, 0.33999999999999997), shade=True, cumulative=False, label='False')
    g2 = sns.kdeplot(selected_data[selected_data["correctness"] == "CORRECT"].distances, ax=ax,
                     color=(0.33999999999999997, 0.43879999999999986, 0.86), shade=True, cumulative=False,
                     label='Correct')

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Cumulative Distance Distributions')
    ax.set(xlabel='Euclidean Distance', ylabel='Probability')

    selected_data = data[data["edge_removal_possibility"] == edge_removal_possibility]
    g1 = sns.kdeplot(selected_data[selected_data["correctness"] == "FALSE"].distances, ax=ax,
                     color=(0.86, 0.5712, 0.33999999999999997), shade=True, cumulative=True, label='False')
    g2 = sns.kdeplot(selected_data[selected_data["correctness"] == "CORRECT"].distances, ax=ax,
                     color=(0.33999999999999997, 0.43879999999999986, 0.86), shade=True, cumulative=True,
                     label='Correct')
