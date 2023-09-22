from sharding_algorithm import shard_allocation
from reputation_recalculation_algorithm import with_decay
import random
import matplotlib.pyplot as plt
import copy



def generate_nodes_list(num_elements, malicious_percentage):
    nodes = []
    num_malicious = int(num_elements * malicious_percentage)

    for i in range(1, num_elements + 1):
        score = 50.0
        is_malicious = i <= num_malicious
        nodes.append({"id": i, "score": score, "malicious": is_malicious})

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


def calculate_the_network(nodes, rep_epoch_num, k, e, malicious_percentage):
    result = []
    new_nodes = copy.deepcopy(nodes)


    for rep_epoch in range(rep_epoch_num):
        shards_formatted = shard_allocation(new_nodes, k, e)

        for shard in shards_formatted:
            for node in shard:
                if node["malicious"]:
                    good = random.randint(85, 95)
                    bad = 100 - good
                    contributions = generate_contributions_array(good, 0, bad)
                else:
                    good = random.uniform(95, 99.9999999999)
                    bad = 100 - good
                    contributions = generate_contributions_array(good, 0, bad)
                    if node["id"] == 100:
                        contributions = generate_contributions_array(99.9, 0, 0.1)
                new_score_decay = with_decay(node["score"], contributions)
                node["score"] = new_score_decay

        # Update the original 'new_nodes' with the scores from 'shards_formatted'
        for new_node in new_nodes:
            for shard in shards_formatted:
                for node in shard:
                    if new_node["id"] == node["id"] and new_node["score"] != node["score"]:
                        new_node["score"] = node["score"]


        result.append({"epoch_num": rep_epoch, "shards": shards_formatted, "malicious_percentage": malicious_percentage})

    return result


# Function to calculate the accumulated reputation score for each shard
def calculate_accumulated_scores(epoch_data, epochs):
    accumulated_scores = {}
    epochs = epochs
    for epoch in epoch_data:
        epoch_num = epoch['epoch_num']
        if epoch_num in epochs:
            for i, shard_data in enumerate(epoch['shards']):
                shard_id = f"Shard {i+1}"  # Use i+1 as the shard ID
                scores = [node['score'] for node in shard_data]
                accumulated_score = sum(scores)
                if shard_id not in accumulated_scores:
                    accumulated_scores[shard_id] = {epoch_num: accumulated_score}
                else:
                    accumulated_scores[shard_id][epoch_num] = accumulated_score
    return accumulated_scores


def plot_bar_chart(epoch_data, accumulated_scores, epoch_num, ax):
    shards = list(accumulated_scores.keys())
    scores = [accumulated_scores[shard].get(epoch_num, 0) for shard in shards]

    malicious_distribution = calculate_malicious_distribution(epoch_data)
    malicious_percentages = [malicious_distribution[epoch_num][f'Shard {i+1}']['num_malicious_percent'] for i in range(len(shards))]

    # Create two separate y-axes
    ax1 = ax
    ax2 = ax.twinx()

    ax1.bar(shards, scores, color='green', edgecolor='black', label='Accumulated Score')
    ax2.bar(shards, malicious_percentages, color='red', edgecolor='black', label='Malicious Percent')

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

    ax1.set_ylim(0, 550)

    ax2.set_ylim(0, 100)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4)


def calculate_malicious_distribution(epoch_data):
    shard_info = {}

    for epoch in epoch_data:
        epoch_num = epoch['epoch_num']
        shard_info[epoch_num] = {}
        total_nodes = sum(len(shard) for shard in epoch['shards'])
        malicious_percentage = epoch['malicious_percentage']
        shard_percentages = []

        for i, shard in enumerate(epoch['shards']):
            shard_id = f"Shard {i+1}"
            num_malicious = len([node for node in shard if node["malicious"]])
            percent_malicious = (num_malicious / len(shard)) * 100 if len(shard) > 0 else 0
            shard_percentages.append(percent_malicious)

            shard_info[epoch_num][shard_id] = {
                'num_malicious_percent': percent_malicious,
                'num_malicious_number': num_malicious,
                'contribution_to_total': 0.0,  # Initialize the contribution for now
            }

        # Calculate the total malicious percentage of all shards for this epoch
        total_shard_percentage = sum(shard_percentages)

        # Update the 'contribution_to_total' field for each shard if total_shard_percentage is not zero
        if total_shard_percentage != 0:
            for shard_id in shard_info[epoch_num]:
                shard_info[epoch_num][shard_id]['contribution_to_total'] = (shard_info[epoch_num][shard_id]['num_malicious_percent'] / total_shard_percentage) * malicious_percentage

    return shard_info


def testing2(number_of_nodes, malicious_percentage, rep_epoch_num, epochs, k, e):

    nodes = generate_nodes_list(number_of_nodes, malicious_percentage)

    data = calculate_the_network(nodes, rep_epoch_num, k, e, malicious_percentage)

    accumulated_scores = calculate_accumulated_scores(data, epochs)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    plot_bar_chart(data, accumulated_scores, 0, axes[0])
    axes[0].set_title('Epoch 0')
    axes[0].set_xlabel('Shard ID')
    axes[0].set_ylabel('Accumulated Reputation Score')

    plot_bar_chart(data, accumulated_scores, 50, axes[1])
    axes[1].set_title('Epoch 50')
    axes[1].set_xlabel('Shard ID')
    axes[1].set_ylabel('Accumulated Reputation Score')

    plot_bar_chart(data, accumulated_scores, 99, axes[2])
    axes[2].set_title('Epoch 100')
    axes[2].set_xlabel('Shard ID')
    axes[2].set_ylabel('Accumulated Reputation Score')

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(f"Malicious percentage: {malicious_percentage}")
    plt.show()


testing2(100, 0.49, 100, [0, 50, 99], 10, 12345)
