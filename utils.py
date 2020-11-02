import numpy as np
import random


def policy_loss(u_sa, mdp_env, opt_u_sa=None):
    '''Compute policy loss wrt opt_u_sa'''
    if opt_u_sa is None:
        #compute optimal policy first
        opt_u_sa = mdp.solve_mdp_lp(mdp_env)
    return np.dot(opt_u_sa, mdp_env.r_sa) - np.dot(u_sa, mdp_env.r_sa)

def sample_l2_ball(dim):
    #generate random normal sample and then normalize to have L2 norm of 1
    sample = np.random.randn(dim)
    return sample / np.linalg.norm(sample)

def write_line(array_list, file_writer):
#write out line to file comma delimited with new line at the end
    f = file_writer
    u_sa = array_list
    for i,sa in enumerate(u_sa):
        if i < len(u_sa) - 1:
            f.write("{},".format(sa))
        else:
            f.write("{}\n".format(sa))

def get_optimal_policy_from_usa(u_sa, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    opt_stoch_pi = np.zeros((num_states, num_actions))
    for s in range(num_states):
        #compute the total occupancy for that state across all actions
        s_tot_occupancy = np.sum(u_sa[s::num_states])
        for a in range(num_actions):
            opt_stoch_pi[s][a] = u_sa[s+a*num_states] / max(s_tot_occupancy, 0.00000001)
    return opt_stoch_pi


def print_stochastic_policy_action_probs(u_sa, mdp_env):
    opt_stoch = get_optimal_policy_from_usa(u_sa, mdp_env)
    print_stoch_policy(opt_stoch, mdp_env)
    # for s in range(mdp_env.get_num_states()):
    #     action_prob_str = "state {}: ".format(s)
    #     for a in range(mdp_env.get_num_actions()):
    #         action_prob_str += "{} = {:.3f}, ".format(mdp_env.get_readable_actions(a), opt_stoch[s,a])
    #     print(action_prob_str)


def print_stoch_policy(stoch_pi, mdp_env):
    for s in range(mdp_env.get_num_states()):
        action_prob_str = "state {}: ".format(s)
        for a in range(mdp_env.get_num_actions()):
            action_prob_str += "{} = {:.3f}, ".format(mdp_env.get_readable_actions(a), stoch_pi[s,a])
        print(action_prob_str)


def print_policy_occupancies_pretty(u_sa, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    for s in range(num_states):
        action_prob_str = "state {}: ".format(s)
        for a in range(mdp_env.num_actions):
            action_prob_str += "{:.3f}, ".format(u_sa[s+a*num_states])
            
        print(action_prob_str)


def print_q_vals_pretty(q_vals, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    for s in range(num_states):
        q_val_str = "state {}: ".format(s)
        for a in range(mdp_env.num_actions):
            q_val_str += "{:.3f}, ".format(q_vals[s+a*num_states])
            
        print(q_val_str)


def display_onehot_state_features(mdp_env):
    state_features = mdp_env.state_features
    state_features_2d = []
    cnt = 0
    for _ in range(mdp_env.num_rows):
        row_features = ""
        for _ in range(mdp_env.num_cols):
            row_features += "{} \t".format(list(state_features[cnt]).index(1))
            cnt += 1
        print(row_features)
        
    #print_as_grid(state_features_2d, mdp_env)

def print_table_row(vals):
    row_str = ""
    for i in range(len(vals) - 1):
        row_str += "{:0.2} & ".format(vals[i])
    row_str += "{:0.2} \\\\".format(vals[-1])
    return row_str


def logsumexp(x):
    max_x = np.max(x)
    sum_exp = 0.0
    for xi in x:
        sum_exp += np.exp(xi - max_x)
    return max(x) + np.log(sum_exp)

def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def print_policy_from_occupancies(proposal_occupancies, mdp_env):
    policy = get_optimal_policy_from_usa(proposal_occupancies, mdp_env)
    cnt = 0
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            if cnt not in mdp_env.terminals:
                row_str += mdp_env.get_readable_actions(np.argmax(policy[cnt])) + "\t"
            else:
                row_str += ".\t"  #denote terminal with .
            cnt += 1
        print(row_str)

def get_policy_string_from_occupancies(u_sa, mdp_env):
    #get stochastic policy
    opt_stoch = get_optimal_policy_from_usa(u_sa, mdp_env)
    cnt = 0
    policy_string_list = []
    for s in range(mdp_env.num_states):
        if s in mdp_env.terminals:
            policy_string_list.append(".")
        else:
            action_str = ""
            for a in range(mdp_env.num_actions):
                if opt_stoch[s,a] > 0.001:
                    action_str += mdp_env.get_readable_actions(a)
            policy_string_list.append(action_str)
    return policy_string_list


def get_stoch_policy_string_dictionary_from_occupancies(u_sa, mdp_env):
    #get stochastic policy
    opt_stoch = get_optimal_policy_from_usa(u_sa, mdp_env)
    cnt = 0
    policy_string_dictionary_list = []
    for s in range(mdp_env.num_states):
        if s in mdp_env.terminals:
            policy_string_dictionary_list.append({".":1.0})
        else:
            action_prob_dict = {}
            for a in range(mdp_env.num_actions):
                action_prob_dict[mdp_env.get_readable_actions(a)] = opt_stoch[s,a]
            policy_string_dictionary_list.append(action_prob_dict)
    return policy_string_dictionary_list


def print_policy_from_occupancies(u_sa,mdp_env):
    cnt = 0
    policy = get_optimal_policy_from_usa(u_sa, mdp_env)
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            if cnt not in mdp_env.terminals:
                row_str += mdp_env.get_readable_actions(np.argmax(policy[cnt])) + "\t"
            else:
                row_str += ".\t"  #denote terminal with .
            cnt += 1
        print(row_str)


def print_as_grid(x, mdp_env):
    #print into a num_rows by num_cols grid
    cnt = 0
    for r in range(mdp_env.num_rows):
        row_str = ""
        for c in range(mdp_env.num_cols):
            row_str += "{:.2f}\t".format(x[cnt])
            cnt += 1
        print(row_str)


def rollout_from_usa(start, horizon, u_sa, mdp_env):
    #generate a demonstration starting at start of length horizon
    demonstration = []
    #first get the stochastic optimal policy
    policy = get_optimal_policy_from_usa(u_sa,mdp_env)

    #rollout for H steps or until a terminal is reached
    curr_state = start
    #print('start',curr_state)
    steps = 0
    while curr_state not in mdp_env.terminals and steps < horizon:
        #print('actions',policy[curr_state])
        #select an action choice according to policy action probs
        a = np.random.choice(range(mdp_env.num_actions), p = policy[curr_state])
        #print(a)
        demonstration.append((curr_state, a))
        #sample transition
        action_transition_probs = mdp_env.Ps[a][curr_state]
        s_next = np.random.choice(range(mdp_env.num_states), p = action_transition_probs)
        curr_state = s_next
        steps += 1
        #print('next state', curr_state)
    if curr_state in mdp_env.terminals:
        #append the terminal state
        demonstration.append((curr_state, None))  #no more actions available
    return demonstration

def u_sa_from_demos(demos, mdp_env):
    '''takes as input
    demos: a list of trajectories where each trajectory is a list of (s,a) pairs
    mdp_env: the mdp with the relevant features

    for now it assumes that the features are only a function of the state
    '''

    #make sure it is a list of lists of (s,a) pairs
    assert type(demos) == list
    assert type(demos[0]) == list
    assert type(demos[0][0]) == tuple
    assert len(demos[0][0]) == 2

    feature_cnts = np.zeros(mdp_env.get_reward_dimensionality())
    for d in demos:
        for t, s_a in enumerate(d):
            s,a = s_a
            feature_cnts += mdp_env.gamma ** t * mdp_env.state_features[s]
            # print(feature_cnts)

    #normalize by number of demonstrations
    feature_cnts /= len(demos)
    
    return feature_cnts


#TODO: maybe try and do one with just feature weights??
def get_worst_case_state_rewards_ird(posterior, u_E, mdp_env):
    #based on IRD formulation by Dylan
    #subtract the expected feature counts as a baseline
    
    #go through each feature and find the minimum value
    baselined_state_features = mdp_env.state_features - u_E

    #posterior is n x k and state features are num_states x k

    #now take the minimum per state
    state_dists = np.dot(posterior, np.array(baselined_state_features).transpose())

    return np.min(state_dists, axis=0)


def get_worst_case_feature_weights_binary_ird(posterior, u_E, mdp_env):
    #based on IRD formulation by Dylan
    #subtract the expected feature counts as a baseline
    
    #go through each feature and find the minimum value
    baselined_binary_features = np.eye(mdp_env.get_reward_dimensionality()) - u_E

    # print("baselined features", baselined_binary_features)
    # print("posterior")
    # print(posterior)

    #posterior is n x k and state features are num_states x k

    # import matplotlib.pyplot as plt
    # feature_names = ['dirt', 'grass', 'target', 'lava']
    # plt.figure()
    # plt.title("raw posterior")
    # for w in range(len(posterior[0])):
    #     plt.plot(posterior[:,w],label=feature_names[w])
    # plt.legend()
    

    #now take the minimum per state
    w_dists = np.dot(posterior, np.array(baselined_binary_features).transpose())

    # print("w_dists")
    # print(w_dists)

    # print("w sorted")
    # print(np.sort(w_dists,axis=0))

    # plt.figure()
    # plt.title("baselined")
    # for w in range(len(posterior[0])):
    #     plt.plot(w_dists[:,w],label=feature_names[w])
    # plt.legend()
    # plt.show()

    # print()

    min_ws = np.min(w_dists, axis=0)
    print("min ws", min_ws)

    return min_ws


def convert_w_to_rsa(W, mdp_env):
    k = mdp_env.num_states * mdp_env.num_actions
    n,w_dim = W.shape
    

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((n,k))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[i,:] = mdp_env.transform_to_R_sa(W[i,:]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()
    return R


