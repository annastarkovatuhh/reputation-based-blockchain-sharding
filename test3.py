from sharding_algorithm import shard_allocation
from reputation_recalculation_algorithm import with_decay
import random
import matplotlib.pyplot as plt
import copy
import numpy as np


def generate_nodes_list(num_elements, initial_malicious_percentage):
    nodes = []

    num_malicious = int(num_elements * (initial_malicious_percentage / 100))

    malicious_indices = random.sample(range(num_elements), num_malicious)

    for i in range(1, num_elements + 1):
        score = 50
        is_malicious = i in malicious_indices
        nodes.append({"id": i, "score": score, "malicious": is_malicious})

    # Make sure we have exactly the desired number of malicious nodes
    while num_malicious > 0:
        random_node = random.choice(nodes)
        if not random_node["malicious"]:
            random_node["malicious"] = True
            num_malicious -= 1

    return nodes


def print_nodes(nodes):
    for node in nodes:
        print(f"Node {node['id']}: Score = {node['score']}, Malicious = {node['malicious']}")


def print_shards(shards):
    for i, shard in enumerate(shards):
        print(f"Shard {i+1}: {shard}")


def generate_contributions_array(w1, w2, w3):
    contributions_array = []
    for i in range(10):
        contributions = random.choices([1, 0, -1], weights=[w1, w2, w3], k=100)
        contributions_array.append(contributions)
    return contributions_array


def increasing_bounded_function(x):
    return 0.49 * (1 - np.exp(-x/100)) + 0.001 * x


def dynamic_node_behavior(nodes, rep_epoch_num, k, e, max_malicious_percentage):
    new_nodes = copy.deepcopy(nodes)

    malicious_nodes_history = []  # To track malicious nodes added in previous epochs
    results = []  # Store the results of calculate_the_network for each epoch

    for rep_epoch in range(rep_epoch_num):

        if rep_epoch % 10 == 0 and rep_epoch >= 10:

            potential_nodes = [node for node in new_nodes if not node["malicious"]]
            nodes_to_make_malicious = random.sample(potential_nodes, 4)

            for node in nodes_to_make_malicious:
                node["malicious"] = True
                malicious_nodes_history.append(node)

            for node in new_nodes:
                for history_node in malicious_nodes_history:
                    if history_node["id"] == node["id"]:
                        node["malicious"] = history_node["malicious"]


        shards_formatted = shard_allocation(new_nodes, k, e)

        for shard in shards_formatted:
            for node in shard:
                if node["malicious"]:
                    good = random.randint(80, 95)
                    bad = 100 - good
                    contributions = generate_contributions_array(good, 0, bad)
                else:
                    good = random.uniform(95, 99.9999999999)
                    bad = 100 - good
                    contributions = generate_contributions_array(good, 0, bad)
                new_score_decay = with_decay(node["score"], contributions)
                node["score"] = new_score_decay

        # Update the original 'new_nodes' with the scores from 'shards_formatted'
        for new_node in new_nodes:
            for shard in shards_formatted:
                for node in shard:
                    if new_node["id"] == node["id"] and new_node["score"] != node["score"]:
                        new_node["score"] = node["score"]

        for i, shard in enumerate(shards_formatted):
            num_malicious = len([node for node in shard if node['malicious']])
            malicious_percent = num_malicious / len(shard)
            accumulated_score = sum(node["score"] for node in shard)
            shard.append({"malicious_percent": malicious_percent, "accumulated_score": accumulated_score})

        result = {"epoch_num": rep_epoch, "shards": shards_formatted}
        results.append(result)

    return results


def plot_bar_chart(network_results, epoch_num, ax):
    shards = [result for result in network_results if result["epoch_num"] == epoch_num][0]["shards"]
    shards_indexes = list(range(1, len(shards) + 1))
    accumulated_scores = [shard[-1]['accumulated_score'] for shard in shards]
    malicious_percentages = [shard[-1]['malicious_percent']*100 for shard in shards]


    ax1 = ax
    ax2 = ax.twinx()

    ax1.bar(shards_indexes, accumulated_scores, color='green', edgecolor='black', label='Accumulated Score')
    ax2.bar(shards_indexes, malicious_percentages, color='red', edgecolor='black', label='Malicious Percent')

    ax1.set_xticks(range(len(shards)))
    ax1.set_xticklabels(range(len(shards)))

    ax1.set_xlabel('Shards')
    ax1.set_ylabel('Accumulated Reputation Score')
    ax2.set_ylabel('Malicious Percent')
    ax1.set_title(f'Epoch {epoch_num} - Accumulated Reputation Scores and Malicious Percent')

    for bar in ax1.patches:
        bar.set_color('green')

    for bar in ax2.patches:
        bar.set_color('red')

    ax1.set_ylim(0, 650)
    ax2.set_ylim(0, 100)



def testing3(num_nodes, rep_epoch_num, k,e,max_malicious_percentage, epochs_to_plot):

    nodes = generate_nodes_list(num_nodes, 0)
    network_results = dynamic_node_behavior(nodes, rep_epoch_num, k, e, max_malicious_percentage)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    for i, epoch in enumerate(epochs_to_plot):
        row = i // 3
        col = i % 3
        plot_bar_chart(network_results, epoch, axes[row, col])
        axes[row, col].set_title(f'Epoch {epoch}')
        axes[row, col].set_xlabel('Shard ID')
        axes[row, col].set_ylabel('Accumulated Reputation Score')

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(f"Malicious percentage: {0.5}")
    plt.show()


testing3(100,100,10,12345,0.5,[0,10,30,50,70,99])

