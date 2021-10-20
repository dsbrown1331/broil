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

def bimodal_dist(mean1, mean2, prop1, prop2):
  X1 = np.random.normal(mean1, 7.5, prop1)
  X2 = np.random.normal(mean2, 5, prop2)
  X = np.concatenate([X1, X2])
  X = np.clip(X, 0, 90)
  return X

def simple_roads():
    NUM_TRIALS = 100
    states = ["a", "b", "c"]
    roads = [("a", "b"), ("b" "c")]
    freeways = [("a", "c")]
    actions = roads + freeways
    transitions = np.zeros((len(actions), len(states), len(states)))
    for i, action in enumerate(actions):
        start, end = action
        for j, state in enumerate(states):
            if state == start:
                transitions[i][j][states.index(end)] = 1
            else:
                transitions[i][j][j] = 1

    r_sa = r = np.ndarray(shape=(len(states) * len(actions), NUM_TRIALS))
    i = 0
    for s in states:
        for start, end in roads:
            if s == start:
                r[i] = np.random.normal(45, 7.5, NUM_TRIALS)
            else:
                r[i] = np.zeros(NUM_TRIALS)
            i += 1
        for start, end in freeways:
            if s == start:
                r[i] = bimodal_dist(75, 20, 75, 25)
            else:
                r[i] = np.zeros(NUM_TRIALS)
            i += 1

    gamma = 0.99
    init_dist = np.zeros(len(states))
    mdp_env = mdp.roadsMDP(states, actions, r_sa, transitions, gamma, init_dist)

    return mdp_env, r_sa


mdp_env, r_sa = simple_roads()
u_expert = np.zeros(mdp_env.get_num_actions() * mdp_env.get_num_states())
posterior_probs = np.ones(100) / 100
alpha = 0.01
lamda = 0.05
debug = True
robust_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, r_sa, posterior_probs, alpha, debug, lamda)
print("=" * 100)
print(robust_opt_usa[:3])
print(max(robust_opt_usa[:3]))
