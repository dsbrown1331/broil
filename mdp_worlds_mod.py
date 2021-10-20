import utils
import mdp
import numpy as np

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
    X1 = np.random.normal(mean1, 2, int(prop1 * num_trials))
    X2 = np.random.normal(mean2, 2, int((1 - prop1) * num_trials))
    X = np.concatenate([X1, X2])
    return X

def simple_roads():
    NUM_TRIALS = 10
    states = ["a", "b", "c"]
    roads = [("a", "b"), ("b" "c")]
    freeways = [("a", "c")]
    actions = roads + freeways
    transitions = np.zeros((len(actions), len(states), len(states)))
    # add bidirectional roads
    # define a cost function
    # check for terminals
    # add check for type of road in reward func
    for i, action in enumerate(actions):
        start, end = action
        for j, state in enumerate(states):

            if state == start:
                transitions[i][j][states.index(end)] = 1
            elif state == end:
                transitions[i][states.index(end)][states.index(start)] = 1
            else:
                transitions[i][j][j] = 1
    for i, action in enumerate(actions):
        j = len(states) - 1
        transitions[i][j] = np.zeros(len(states))
        transitions[i][j][j] = 1
    print(transitions)

    r_sa = r = np.ndarray(shape=(len(states) * len(actions), NUM_TRIALS))
    i = 0
    # add a big penalty for self transition.
    # for r_sa iterate through actions then states
    for start, end in roads:
        for s in states:
            if s == start:
                r[i] = np.random.normal(30, 2, NUM_TRIALS)
            else:
                r[i] = np.zeros(NUM_TRIALS) - 10 ** 4
            i += 1
    for start, end in freeways:
        for s in states:
            if s == start:
                r[i] = bimodal_dist(-10, -40, 0.7, NUM_TRIALS)
            else:
                r[i] = np.zeros(NUM_TRIALS) - 10 ** 4
            i += 1

    gamma = 0.99
    init_dist = np.array([1, 0, 0])
    mdp_env = mdp.roadsMDP(states, actions, r_sa, transitions, gamma, init_dist)
    print(r_sa)
    return mdp_env, r_sa



mdp_env, r_sa = simple_roads()
u_expert = np.zeros(mdp_env.get_num_actions() * mdp_env.get_num_states())
"""
if we had expert data we could feed this in to CVAR
Try to be baseline robust -> something different to the demonstrator would have more risk
Appendix from the paper is most useful
"""
posterior_probs = np.ones(10) / 10
for lamda in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
    alpha = 0.95
    debug = False
    robust_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_sa, posterior_probs, alpha, debug, lamda)
    print("=" * 100)
    print(robust_opt_usa[::3])
    print(np.argmax(robust_opt_usa[::3]))

"""
More complex road network
visualization of data - Maybe coordinates

"""
