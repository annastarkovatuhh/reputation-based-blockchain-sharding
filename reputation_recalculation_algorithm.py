import math

# for reward/penalty calculation
maxRewardPoints = 0.005
basePenaltyPoints = 0.1
epochsNum = 10


# Define the decay function for time-based weights
# more recent behaviour has more weight than the old one
def decay_function(position):
    # exponential decay
    decay_factor = 0.1
    return math.exp(-decay_factor * position)


# score calculation function
def recalculate_score(score, contributions):
    reputation_score = score
    reward_points = maxRewardPoints * (1 - reputation_score/100)  # Decreasing function for reward points
    penalty_points = basePenaltyPoints * (reputation_score / 100)

    num_valid = contributions.count(1)
    num_malicious = contributions.count(-1)

    reward_points *= num_valid
    penalty_points *= num_malicious

    reputation_score += reward_points
    reputation_score -= penalty_points

    return reputation_score


def with_decay(score, contributions_list):
    scores = []

    current_score = score

    for contributions in contributions_list:

        new_score = recalculate_score(current_score, contributions)
        scores.insert(0, new_score)
        current_score = new_score


    weighted_score_sum = 0
    total_weight = 0
    weights = []
    weighted_scores = []

    for position, score in enumerate(scores, start=1):
        weight = decay_function(position)
        weights.append(weight)
        weighted_score = score * weight
        weighted_scores.append(weighted_score)

        weighted_score_sum += weighted_score
        total_weight += weight

    reputation_score = weighted_score_sum / total_weight if total_weight != 0 else 0

    new_score_decay = reputation_score

    return new_score_decay


def no_decay(score, contributions_list):

    current_score = score

    for contributions in contributions_list:

        new_score = recalculate_score(current_score, contributions)
        current_score = new_score

    return current_score


