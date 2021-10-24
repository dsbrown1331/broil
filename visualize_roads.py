import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap


def get_location_of_state(f, t, states):
    return states[[i[0] for i in states].index(f)][1], states[[i[0] for i in states].index(t)][1]

def norm_action_prob(states, actions, i, robust_opt_usa):
    f = actions[i][0]
    state_index = [i[0] for i in states].index(f)
    all_actions_from_state = np.sum(robust_opt_usa[state_index::len(actions)])
    curr_prob = np.sum(robust_opt_usa[state_index + i * len(actions)])
    return curr_prob / all_actions_from_state

def create_plot(states, actions, opt_policy):
    circle_size = 0.05
    start_color = 'b'
    end_color = 'b'
    node_color = 'g'

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot

    for i, action in enumerate(actions):
        prob = norm_action_prob(states, actions, i, opt_policy)
        print(prob)
        f, t = get_location_of_state(action[0], action[1], states)
        ax.plot([f[0], t[0]], [f[1], t[1]], color=(0.5, 0.5, 1 - (1 * prob)), linewidth=2, alpha = 1)



    for state in states:
        loc = state[1]
        if state == states[0]:
            circle = plt.Circle(loc, circle_size, color=start_color, alpha = 1)
        elif state == states[-1]:
            circle = plt.Circle(loc, circle_size, color=end_color, alpha = 1)
        else:
            circle = plt.Circle(loc, circle_size, color=node_color, alpha = 1)
        ax.add_patch(circle)

    ax.set(xlim=(0, 1), ylim=(0.25, 1))
    ax.set_aspect( 1 )
    plt.show(block=False)
    plt.pause(1)
    plt.close()



# if __name__== "__main__":
