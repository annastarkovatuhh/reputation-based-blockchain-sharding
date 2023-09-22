import matplotlib.pyplot as plt
import random
import math
from reputation_recalculation_algorithm import with_decay, no_decay



figure, axis = plt.subplots(2, 3, figsize=(15, 10))


def generate_contributions_array(w1, w2, w3, maliciousness):
    contributions_array = []
    for i in range(10):
        if maliciousness > 0.5:
            contributions = random.choices([1, 0, -1], weights=[w1, w2, w3], k=100)
        else:
            contributions = random.choices([1, 0, -1], weights=[w1, w2, w3], k=100)
        contributions_array.append(contributions)
    return contributions_array



def gradual_malicious_behavior(num_epochs, malicious_start_epoch, decay_rate=0.95):
    if num_epochs <= malicious_start_epoch:
        return 1  # Trustworthy behavior
    else:
        return decay_rate  # Gradually become malicious


def gradual_malicious_behavior2(num_epochs, malicious_start_epoch):
    if num_epochs <= malicious_start_epoch:
        return 1  # Trustworthy behavior
    else:
        return 0.5 + 0.5 * math.sin(0.1 * (num_epochs - malicious_start_epoch))  # Alternating behavior



def testing(initial_score, rep_epochs_num, malicious_start_epoch, type, w1, w2, w3, place):
    x_iterate_values = []
    y_score_values = []
    y_decay_score_values = []

    node = {"score": initial_score, "score_decay": initial_score}

    for i in range(rep_epochs_num):
        if type == 'recent_positive':
            maliciousness = gradual_malicious_behavior(rep_epochs_num, malicious_start_epoch, 0.3)
            if i <= malicious_start_epoch:
                contributions = generate_contributions_array(w1, w2, w3, maliciousness)
            else:
                contributions = generate_contributions_array(90, 0, 10, maliciousness)

        elif type == 'recent_negative':
            maliciousness = gradual_malicious_behavior(rep_epochs_num, malicious_start_epoch, 0.3)
            if i <= malicious_start_epoch:
                contributions = generate_contributions_array(w1, w2, w3, maliciousness)
            else:
                contributions = generate_contributions_array(w3, w2, w1, maliciousness)

        elif type == 'gradual_malicious':
            maliciousness = gradual_malicious_behavior(rep_epochs_num, malicious_start_epoch, 0.5)
            if maliciousness > 0.5:
                contributions = generate_contributions_array(70, 5, 25, maliciousness)
            else:
                contributions = generate_contributions_array(w1, w2, w3, maliciousness)

        new_score = no_decay(node["score"], contributions)
        new_score_decay = with_decay(node["score_decay"], contributions)
        x_iterate_values.append(i)
        y_score_values.append(new_score)
        y_decay_score_values.append(new_score_decay)
        node["score"] = new_score
        node["score_decay"] = new_score_decay

    axis[place].plot(x_iterate_values, y_score_values, 'r')
    axis[place].plot(x_iterate_values, y_decay_score_values, 'g')
    axis[place].set_xlabel("epoch number")
    axis[place].set_ylabel("reputation score")
    axis[place].set_title(type)
    axis[place].legend(['noDecay', 'withDecay'])


# Zooming function
def apply_zoom(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


# Test 1 - Recent Positive Normal
testing(50, 100, 10, 'recent_positive', 20, 5, 75, (0, 0))

# Test 2 - Recent Negative Normal
testing(50, 100, 70, 'recent_negative', 90, 5, 5, (0, 1))

# Test 3 - Gradual Malicious Normal
testing(50, 100, 40, 'gradual_malicious', 90, 5, 5, (0, 2))

# Test 4 - Recent Positive Zoomed
axis[1, 0].clear()  # Clear the existing plot
lines = axis[0, 0].lines
for line in lines:
    x, y = line.get_xydata().T
    axis[1, 0].plot(x, y, color=line.get_color(), label=line.get_label())
apply_zoom(axis[1, 0], 0, 60, 0, 30)  # Apply zoom to the plot
axis[1,0].set_xlabel("epoch number")
axis[1,0].set_ylabel("reputation score")
axis[1, 0].set_title('recent_positive (Zoomed)')
axis[1, 0].legend(['noDecay', 'withDecay'])

# Test 5 - Recent Negative Zoomed
axis[1, 1].clear()  # Clear the existing plot
lines = axis[0, 1].lines
for line in lines:
    x, y = line.get_xydata().T
    axis[1, 1].plot(x, y, color=line.get_color(), label=line.get_label())
apply_zoom(axis[1, 1], 50, 90, 0, 45)  # Apply zoom to the plot
axis[1,1].set_xlabel("epoch number")
axis[1,1].set_ylabel("reputation score")
axis[1, 1].set_title('recent_negative (Zoomed)')
axis[1, 1].legend(['noDecay', 'withDecay'])

# Test 6 - Gradual Malicious Zoomed
axis[1, 2].clear()  # Clear the existing plot
lines = axis[0, 2].lines
for line in lines:
    x, y = line.get_xydata().T
    axis[1, 2].plot(x, y, color=line.get_color(), label=line.get_label())
apply_zoom(axis[1, 2], 0, 60, 30, 50)  # Apply zoom to the plot
axis[1,2].set_xlabel("epoch number")
axis[1,2].set_ylabel("reputation score")
axis[1, 2].set_title('gradual_malicious (Zoomed)')
axis[1, 2].legend(['noDecay', 'withDecay'])


plt.tight_layout()
plt.show()