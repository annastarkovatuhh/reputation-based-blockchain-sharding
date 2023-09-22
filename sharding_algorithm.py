import random
import copy


def shard_allocation(nodes, k, e):
    # Initialize shards and random number seed generator
    S = [list() for _ in range(k)]  # Store nodes as lists instead of sets
    G_rand = random.Random(e)

    # Sort the nodes based on the "score" field in descending order
    nodes_sorted = sorted(nodes, key=lambda x: x["score"], reverse=True)

    for node in nodes_sorted:
        # Find the shard(s) with the lowest number of nodes
        min_shard_length = min(len(s) for s in S)
        min_shards = [i for i, s in enumerate(S) if len(s) == min_shard_length]

        # If there's only one shard with the lowest number of nodes, assign the node to it
        if len(min_shards) == 1:
            S[min_shards[0]].append(copy.deepcopy(node))  # Append a deep copy of the node dictionary to the shard
        else:
            # If there are multiple shards with the same lowest number of nodes, choose one randomly
            shard_idx = G_rand.choice(min_shards)
            S[shard_idx].append(copy.deepcopy(node))  # Append a deep copy of the node dictionary to the shard

    return S

