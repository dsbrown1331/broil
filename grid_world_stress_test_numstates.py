#!/usr/bin/env python
# coding: utf-8

# Code for running experiment where a feature is present in test env but not in train. Also shows that our method scales up to larger grid domains.


import mdp
import mdp_worlds
import utils
import numpy as np
import random 
import bayesian_irl
import plot_gridworld as pg



#create a train domain

init_seed = 12345
np.random.seed(init_seed)
random.seed(init_seed)

demo_horizon = 100
num_demos = 1


num_rows = 10
num_cols = 10
num_features = 6


train_mdp = mdp_worlds.negative_sideeffects_goal(num_rows, num_cols, num_features, unseen_feature=False)
train_mdp.set_reward_fn(np.array([-.1,-.6,-.1,-0.6,-2,0]))
opt_sa_train = mdp.solve_mdp_lp(train_mdp)
print("===========================")
print("Training MDP with No Lava")
print("===========================")

print("Optimal Policy")
utils.print_policy_from_occupancies(opt_sa_train, train_mdp)
print("reward")
utils.print_as_grid(train_mdp.r_s, train_mdp)
print("features")
utils.display_onehot_state_features(train_mdp)


import numpy as np
np.random.randint(60)


init_demo_states = [0,9,90,99]#mdp_env.num_cols * (mdp_env.num_rows - 1)
traj_demonstrations = []
demo_set = set()
for d in range(num_demos):
    # np.random.seed(init_seed + d)
    # random.seed(init_seed + d)
    for s in init_demo_states:
        #s = init_demo_state #mdp_env.init_states[0] # only one initial state
        demo = utils.rollout_from_usa(s, demo_horizon, opt_sa_train, train_mdp)
        print("demo", d, demo)
        traj_demonstrations.append(demo)
        for s_a in demo:
            demo_set.add(s_a)
demonstrations = list(demo_set)
print("demonstration")
print(demonstrations)

state_feature_list = [tuple(fs) for fs in train_mdp.state_features]

#Now let's run Bayesian IRL on this demo in this mdp with a placeholder feature to see what happens.
beta = 10.0
step_stdev = 0.05

num_samples = 1000
mcmc_norm = "inf"
likelihood = "birl"


birl = bayesian_irl.BayesianIRL(train_mdp, beta, step_stdev, debug=False, mcmc_norm=mcmc_norm, likelihood=likelihood, prior="non-pos")

map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, num_samples, True)




print(train_mdp.feature_weights)



burn = 200
skip = 10
r_chain_burned = r_chain[burn::skip]

u_expert = utils.u_sa_from_demos(traj_demonstrations, train_mdp)
expert_returns = np.sort(np.dot(r_chain_burned, u_expert))


#get the r_sa matrix from the posterior 
Rsa = utils.convert_w_to_rsa(r_chain_burned, train_mdp)



#what does the BROIL policy do?


#create test MDP
num_rows = 60
num_cols = 60
num_features = 6
num_reps = 20

init_seed = 12345
np.random.seed(init_seed)
random.seed(init_seed)

test_mdp = mdp_worlds.negative_sideeffects_goal(num_rows, num_cols, num_features, unseen_feature=True)
#opt_sa_test = mdp.solve_mdp_lp(test_mdp)


#what does the BROIL policy do?
#Now let's see what CVaR optimization does.t = time.time()

import time


lamda = 0.0
alpha = 0.95



debug = False


n = r_chain_burned.shape[0]
print("num reward hypothesis", n)
posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC
traj_demonstrations = [demonstrations]
u_expert = utils.u_sa_from_demos(traj_demonstrations, train_mdp)
print("u expert", u_expert)



run_times = []
for rep in range(num_reps):
    print(rep)    
    
    t = time.time()
    regret_opt_usa, cvar_value, exp_ret = mdp.solve_max_cvar_policy(test_mdp, u_expert, r_chain_burned.transpose(), posterior_probs, alpha, debug, lamda)
    #utils.print_stochastic_policy_action_probs(cvar_opt_usa, test_mdp_A)
    elapsed = time.time() - t
    run_times.append(elapsed)
    print(elapsed)
#save run times
# 
import os 
if not os.path.exists('./results/stress_test/'):
    os.makedirs('./results/stress_test/')
np.savetxt("./results/stress_test/grid_world_60_60.csv", run_times, delimiter=",")     





