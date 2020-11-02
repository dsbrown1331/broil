import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plot_dashed_arrow(state, width, ax, direction):
    #print("plotting dashed arrow", direction)
    h_length = 0.15
    shaft_length = 0.4
    
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    #print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=None, head_length=None, fc='k', ec='k',linewidth=4, linestyle=':',fill=False) 
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    
    #print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = h_length
        y_coord += shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -h_length
        y_coord += -shaft_length + h_length
    elif direction is 'left':
        x_end = -h_length
        y_end = 0
        x_coord += -shaft_length + h_length
    elif direction is 'right':
        x_end = h_length
        y_end = 0
        x_coord += shaft_length - h_length
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length, fc='k', ec='k',linewidth=4, fill=False,length_includes_head = True) 

def plot_arrow(state, width, ax, direction, prob=None):
    #print("plotting arrow", direction)
    # if prob:
    #     h_length = 0.15 * min(prob*4,1)
    #     shaft_length = 0.4 * min(prob*4,1)
    # else:
    h_length = 0.15
    shaft_length = 0.4

    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    #print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    # print(x_end, y_end)
    if prob is not None:
        if prob > 0.005:
            ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2*prob, head_length=h_length*prob, fc='k', ec='k',linewidth=4*prob) 
    else:
        ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length, fc='k', ec='k',linewidth=4) 

def plot_dot(state, width, ax):
    ax.plot(state % width, state // width, 'ko',markersize=10)


#plot a stochastic policy and make arrows reflect the probability of action
def plot_optimal_policy_stochastic(pi_stoch, feature_list, num_rows, num_cols, filename = None):
    #reformat into rows and cols
    pi = []
    feature_mat = []
    cnt = 0
    for r in range(num_rows):
        pi_row = []
        f_row = []
        for c in range(num_cols):
            pi_row.append(pi_stoch[cnt])
            f_row.append(feature_list[cnt])
            cnt += 1
        pi.append(pi_row)
        feature_mat.append(f_row)
    
    plot_stochastic_policy(pi, feature_mat, filename)



def plot_optimal_policy_lists(pi_list, feature_list, num_rows, num_cols, filename = None):
    #reformat into rows and cols
    pi = []
    feature_mat = []
    cnt = 0
    for r in range(num_rows):
        pi_row = []
        f_row = []
        for c in range(num_cols):
            pi_row.append(pi_list[cnt])
            f_row.append(feature_list[cnt])
            cnt += 1
        pi.append(pi_row)
        feature_mat.append(f_row)
    
    plot_optimal_policy(pi, feature_mat, filename)


def get_policy_string_from_trajectory(traj, feature_list, mdp_env, filename = None):
    pi_list = ["" for s in range(mdp_env.num_states)]
    #go through trajectory and fill in entries
    for s,a in traj:
        if a is None:
            pi_list[s] = "."
        else:
            pi_list[s] = mdp_env.get_readable_actions(a)
    
    plot_optimal_policy_lists(pi_list, feature_list, mdp_env.num_rows, mdp_env.num_cols, filename)


def plot_stochastic_policy(pi, feature_mat, filename = None):
    plt.figure()

    ax = plt.axes() 
    count = 0
    rows,cols = len(pi), len(pi[0])
    for line in pi:
        for el_dict in line:
            #print("optimal action", el)
            # could be a stochastic policy with more than one optimal action
            for char in el_dict:
                char_prob = el_dict[char]
                #print(char)
                if char is "^":
                    plot_arrow(count, cols, ax, "up", prob=char_prob)
                elif char is "v":
                    plot_arrow(count, cols, ax, "down", prob=char_prob)
                elif char is ">":
                    plot_arrow(count, cols, ax, "right", prob=char_prob)
                elif char is "<":
                    plot_arrow(count, cols, ax, "left", prob=char_prob)
                elif char is ".":
                    plot_dot(count, cols, ax)
                elif el is "w":
                    #wall
                    pass
            count += 1

    
    #use for wall states
    #if walls:
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    
    #mat =[[0,0],[2,2]]
    feature_set = set()
    for mrow in mat:
        for m in mrow:
            feature_set.add(m)
    num_features = len(feature_set)
    print(mat)
    all_colors = ['black','white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan']
    colors_to_use = []
    for f in range(9):#hard coded to only have 9 features right now
        if f in feature_set:
            colors_to_use.append(all_colors[f])
    cmap = colors.ListedColormap(colors_to_use)
    # else:
    #     mat = [[fvec.index(1) for fvec in row] for row in feature_mat]
    #     cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    
    #input()
    
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)



def plot_optimal_policy(pi, feature_mat, filename = None):
    plt.figure()

    ax = plt.axes() 
    count = 0
    rows,cols = len(pi), len(pi[0])
    for line in pi:
        for el in line:
            #print("optimal action", el)
            # could be a stochastic policy with more than one optimal action
            for char in el:
                #print(char)
                if char is "^":
                    plot_arrow(count, cols, ax, "up")
                elif char is "v":
                    plot_arrow(count, cols, ax, "down")
                elif char is ">":
                    plot_arrow(count, cols, ax, "right")
                elif char is "<":
                    plot_arrow(count, cols, ax, "left")
                elif char is ".":
                    plot_dot(count, cols, ax)
                elif el is "w":
                    #wall
                    pass
            count += 1

    
    #use for wall states
    #if walls:
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    
    #mat =[[0,0],[2,2]]
    feature_set = set()
    for mrow in mat:
        for m in mrow:
            feature_set.add(m)
    num_features = len(feature_set)
    print(mat)
    all_colors = ['black','white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan']
    colors_to_use = []
    for f in range(9):#hard coded to only have 9 features right now
        if f in feature_set:
            colors_to_use.append(all_colors[f])
    cmap = colors.ListedColormap(colors_to_use)
    # else:
    #     mat = [[fvec.index(1) for fvec in row] for row in feature_mat]
    #     cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    
    #input()
    
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)

    
def plot_test_query(state, better_action, worse_action, feature_mat, equal_pref = False):

    plt.figure()
    ax = plt.axes() 
    count = 0
    rows,cols = len(feature_mat), len(feature_mat[0])
    if better_action is "^":
        plot_arrow(state, cols, ax, "up")
    elif better_action is "v":
        plot_arrow(state, cols, ax, "down")
    elif better_action is ">":
        plot_arrow(state, cols, ax, "right")
    elif better_action is "<":
        plot_arrow(state, cols, ax, "left")
        
    if equal_pref:
        if worse_action is "^":
            plot_arrow(state, cols, ax, "up")
        elif worse_action is "v":
            plot_arrow(state, cols, ax, "down")
        elif worse_action is ">":
            plot_arrow(state, cols, ax, "right")
        elif worse_action is "<":
            plot_arrow(state, cols, ax, "left")

    
    else:
    
        if worse_action is "^":
            plot_dashed_arrow(state, cols, ax, "up")
        elif worse_action is "v":
            plot_dashed_arrow(state, cols, ax, "down")
        elif worse_action is ">":
            plot_dashed_arrow(state, cols, ax, "right")
        elif worse_action is "<":
            plot_dashed_arrow(state, cols, ax, "left")

    
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    print(mat)
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    cmap = colors.ListedColormap(['black','white','tab:red', 'tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    #ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.show()

    
if __name__=="__main__":
    pi = [['', '^><','.'],['<>v','<','>'],['<>^v','v' ,'^']]
    feature_mat = [[(1,0),(0,1),(0,1)],[(0,1),(0,1),(0,1)],[(0,1), (0,1),(1,0)]  ]      
    plot_optimal_policy(pi, feature_mat)
    
  
    pi = ['v', '^><','.','<>v','<','>','<>^v','v' ,'^']
    feature_mat = [(1,0),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1), (0,1),(1,0)  ]      
    plot_optimal_policy_lists(pi, feature_mat, 3,3)