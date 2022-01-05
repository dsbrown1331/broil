import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cmx
import matplotlib.colors as colors


def get_location_of_state(f, t, states):
    return states[[i[0] for i in states].index(f)][1], states[[i[0] for i in states].index(t)][1]

def norm_action_prob(states, actions, i, robust_opt_usa):
    f = actions[i][0]
    state_index = [i[0] for i in states].index(f)
    all_actions_from_state = np.sum(robust_opt_usa[state_index::len(states)])
    curr_prob = np.sum(robust_opt_usa[state_index + i * len(states)])
    return curr_prob / all_actions_from_state

def create_plot(states, actions, opt_policy, roads, lamda):
    circle_size = 0.025
    start_color = 'b'
    end_color = 'r'
    node_color = 'black'

    fig, ax = plt.subplots(figsize = (10, 10)) # note we must use plt.subplots, not plt.subplot

    jet = cm = plt.get_cmap('summer')
    cNorm  = colors.Normalize(vmin=-0.2, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=jet)


    for i, action in enumerate(actions):
        prob = norm_action_prob(states, actions, i, opt_policy)
        f, t = get_location_of_state(action[0], action[1], states)
        if action in roads:
            ax.plot([f[0], t[0]], [f[1], t[1]], color=scalarMap.to_rgba(1-prob), linewidth=2)
        else:
            ax.plot([f[0], t[0]], [f[1], t[1]], color=scalarMap.to_rgba(1-prob), linewidth=6)


    for state in states:
        loc = state[1]
        if state == states[0]:
            circle = plt.Circle(loc, circle_size, color=start_color, zorder=10)
        elif state == states[-1]:
            circle = plt.Circle(loc, circle_size, color=end_color, zorder=10)
        else:
            circle = plt.Circle(loc, circle_size, color=node_color, zorder=10)
        ax.add_patch(circle)

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect( 1 )
    plt.title("lambda = {}".format(lamda))
    plt.savefig('{}.png'.format(lamda))
    plt.show()
    plt.pause(2)
    plt.close()
