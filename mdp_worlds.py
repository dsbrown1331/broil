import utils
import mdp
import numpy as np
from visualize_roads import *
from astar import *

np.random.seed(1)
def lava_ambiguous_corridor():
    num_rows = 3
    num_cols = 5
    num_states = num_rows * num_cols
    white = (1,0)
    red = (0,1)
    state_features = np.array([white, white, white, white, white,
                               white, red, red, red, red,
                               white, white, white, white, white])
    weights = np.array([-0.1, -0.9])#np.array([-0.26750391, -0.96355677])#np.array([-.18, -.82])
    weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.99
    init_dist = np.zeros(num_states)
    # init_states = [5,4]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)
    term_states = [14]
    init_dist = 1/(num_states) * np.ones(num_states)
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def negative_sideeffects_goal(num_rows, num_cols, num_features, unseen_feature=False):
    #no terminal random rewards and features

    num_states = num_rows * num_cols

    if unseen_feature:
        assert(num_features >=3)

    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last two states randomly (last state is goal, second to last state is possibly unseen)
        if unseen_feature:
            r_idx = np.random.randint(num_features-1)
        else:
            r_idx = np.random.randint(num_features-2)
        state_features.append(features[r_idx])

    #select goal
    goal_state = np.random.randint(num_states)
    state_features[goal_state] = features[-1]


    state_features = np.array(state_features)


    #sample from L2 ball
    weights = -np.random.rand(num_features)
    #set goal as positive
    weights[-1] = +2
    #set unseen as negative
    weights[-2] = -2
    weights = weights / np.linalg.norm(weights)

    print("weights", weights)
    gamma = 0.99
    #let's look at all starting states for now
    init_dist = np.ones(num_states) / num_states
    # init_states = [10]
    # for si in init_states:
    #     init_dist[si] = 1.0 / len(init_states)

    #no terminal
    term_states = [goal_state]

    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env

def bimodal_dist(mean1, mean2, prop1, num_trials):
    assert int(prop1 * num_trials) + int((1 - prop1) * num_trials) == num_trials
    X1 = np.random.normal(mean1, 7.5, int(prop1 * num_trials))
    X2 = np.random.normal(mean2, 5, int((1 - prop1) * num_trials))
    X = np.concatenate([X1, X2])
    return X

def simple_roads():
    NUM_TRIALS = 100
    states_with_loc = [["a", (0.1, 0.5)], ["b", (0.5, 0.75)], ["c", (0.5, 0.25)], ["d", (0.6, 0.5)], ["e", (0.9, 0.5)]]
    states = [i[0] for i in states_with_loc]
    roads = [("a", "b"), ("a", "c"), ("c", "b"), ("d", "e"), ("b", "d")]
    freeways = [("a", "d"), ("c", "e")]
    actions = roads + freeways
    transitions = np.zeros((len(actions), len(states), len(states)))
    # add bidirectional roads DONE
    # define a cost function DONE
    # check for terminals DONE
    # add check for type of road in reward func DONE
    for i, action in enumerate(actions):
        start, end = action
        for j, state in enumerate(states):
            if start == states[-1]:
                transition[i][j] = np.zeros(len(states))
                continue
            if state == start:
                transitions[i][j][states.index(end)] = 1
                transitions[i][states.index(end)][j] = 1

            elif state != end:
                transitions[i][j][j] = 1
    r_sa = r = np.ones(shape=(len(states) * len(actions), NUM_TRIALS))
    i = 0
    for start, end in roads:
        for s in states:
            if s == states[-1]:
                r[i] = np.zeros(NUM_TRIALS)
            elif s == start:
                r[i] = np.random.normal(-10, 0.5, NUM_TRIALS)
            elif s == end:
                r[i] = np.random.normal(-10, 0.5, NUM_TRIALS)
            else:
                if r[i][0] == 1:
                    r[i] = np.zeros(NUM_TRIALS) - 10 ** 3
            i += 1
    for start, end in freeways:
        for s in states:
            if s == states[-1]:
                r[i] = np.zeros(NUM_TRIALS)
            elif s == start:
                r[i] = bimodal_dist(-10, -20, 0.7, NUM_TRIALS)
            elif s == end:
                r[i] = bimodal_dist(-10, -20, 0.7, NUM_TRIALS)
            else:
                if r[i][0] == 1:
                    r[i] = np.zeros(NUM_TRIALS) - 10 ** 3
            i += 1
    r_sa = r_sa.round(decimals =2)
    gamma = 0.99
    init_dist = np.array([1, 0, 0, 0, 0])
    mdp_env = mdp.roadsMDP(states, actions, r_sa, transitions, gamma, init_dist)
    # print(r_sa)
    return mdp_env, r_sa, states_with_loc, actions, roads, freeways



mdp_env, r_sa, states, actions, roads, freeways = simple_roads()
u_expert = np.zeros(mdp_env.get_num_actions() * mdp_env.get_num_states())

"""
if we had expert data we could feed this in to CVAR
Try to be baseline robust -> something different to the demonstrator would have more risk
Appendix from the paper is most useful
"""
astar_policy = astar_search(states, actions, r_sa, states[0], states[-1])
alpha = 0.95
debug = True
lamda_range = np.arange(0, 1, 0.01)
posterior_probs = np.ones(100) / 100
for lamda in np.arange(0.1, 1.05, 0.1):
    robust_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_sa, posterior_probs, alpha, debug, lamda)
    print("=" * 100)

    for i in range(mdp_env.get_num_states()):
        print("action from state {}".format(mdp_env.states[i]))
        print(mdp_env.get_readable_actions(np.argmax(robust_opt_usa[i::mdp_env.get_num_states()])))
    create_plot(states, actions, robust_opt_usa, roads, lamda)

def calc_frontier(mdp_env, u_expert, reward_posterior, posterior_probs, lambda_range, alpha, debug=False):
    '''takes an MDP and runs over a range of lambdas to output the expected value and CVaR of the resulting solutions to the LP
        mdp_env: the mdp to run on
        u_expert: the baseline expert to try and beat (set to zeros just to be robust)
        reward_posterior: the reward posterior from B-IRL(already burned and skiped and ready to run in LP)
        posterior_probs: the probabilities of each element in the posterior (uniform if from MCMC)
        lambda_range: a list of lambda values to try
        alpha: the CVaR alpha (risk sensitivity) higher is more risk-sensitive/conservative
    '''

    cvar_exprews = []

    for lamda in lambda_range:
        cvar_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, reward_posterior, posterior_probs, alpha, debug, lamda)
        cvar_exprews.append((cvar_value, exp_ret))
    return cvar_exprews

cvar_rets = calc_frontier(mdp_env, u_expert, r_sa, posterior_probs, lamda_range, alpha, debug=False)
cvar_rets_array = np.array(cvar_rets)

cvar_lpal, expret_lpal = mdp.solve_cvar_expret_fixed_policy(mdp_env, astar_policy, u_expert, r_sa, posterior_probs, alpha, debug=False)


plt.figure()
ax = plt.subplot()
ax.plot(cvar_lpal, expret_lpal, 's',label='repeated astar', color='red')
# plt.title(r"$\alpha = {}$".format(alpha))
ax.plot(cvar_rets_array[:,0], cvar_rets_array[:,1], '-o',label="BROIL",color='green')



unique_pts_lambdas = []
unique_pts = []
for i,pt in enumerate(cvar_rets_array):
    unique = True
    for upt in unique_pts:
        if np.linalg.norm(upt - pt) < 0.00001:
            unique = False
            break
    if unique:
        unique_pts_lambdas.append((pt[0], pt[1], lamda_range[i]))
        unique_pts.append(np.array(pt))


plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Robustness (CVaR)", fontsize=22)
plt.ylabel("Expected Performance", fontsize=22)

plt.legend(loc='best', fontsize=20)
plt.tight_layout()
#plt.show()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()

"""
More complex road network
visualization of data - Maybe coordinates

"""
