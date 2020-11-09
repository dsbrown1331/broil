import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random
import generate_efficient_frontier

def generate_reward_sample(n_states):
    #costs for no-op are -N(0,10-4) except last state that is -N(100,800)
    r_noop = []
    locs = 1/2
    scales = [i*10 for i in range(n_states)]
    for i in range(n_states):
        r_noop.append(-np.random.gamma(locs, scales[i], 1)[0])
    r_noop = np.array(r_noop)
    #print(r_noop)
    
    #costs for repair are -N(130,1) for all but last state where it is -N(130,20)
    r_repair = -100 + -1 * np.random.randn(n_states)
    #print(r_repair)
    return np.concatenate((r_noop, r_repair))

# def generate_reward_sample():
#     #costs for no-op are -N(0,10-4) except last state that is -N(100,800)
#     r_noop = []
#     shapes = [1,1,1,3]
#     scales = [20,30,40,50]
#     for i in range(num_states):
#         r_noop.append(-np.random.gamma(shapes[i], scales[i], 1)[0])
#     r_noop = np.array(r_noop)
#     #print(r_noop)
    
#     #costs for repair are -N(130,1) for all but last state where it is -N(130,20)
#     r_repair = -100 + -1 * np.random.randn(4)
#     #print(r_repair)
#     return np.concatenate((r_noop, r_repair))

# def generate_reward_mean():
#     #costs for no-op are -N(0,10-4) except last state that is -N(100,800)
#     r_noop = []
#     loc = 1/4
#     scales = [20, 40,80,300]
#     for i in range(num_states):
#         r_noop.append(-loc * scales[i])
#     r_noop = np.array(r_noop)
#     #print(r_noop)
    
#     #costs for repair are -N(130,1) for all but last state where it is -N(130,20)
#     r_repair = -100 * np.ones(4)
#     #print(r_repair)
#     return np.concatenate((r_noop, r_repair))

def generate_posterior_samples(num_samples, n_states):
    #print("samples")
    all_samples = []
    for i in range(num_samples):
        r_sample = generate_reward_sample(n_states)
        all_samples.append(r_sample)
        # r_string = ""
        # for r in r_sample:
        #     r_string += "{:.1f}\t".format(r)
        # print(r_string)

    #print("mean of posterior from samples")
    #print(np.mean(all_samples, axis=0))
    #print(generate_reward_mean(3))

    posterior = np.array(all_samples)

    return posterior.transpose()  #each column is a reward sample


if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    
    num_states_list = [10,50,100,200, 500,1000,2000,5000]
    num_reps = 20
    num_samples = 200
    gamma = 0.95
    alpha = 0.95
    lamda = 0.5


    run_times = np.zeros((num_reps, len(num_states_list)))
    for i,num_states in enumerate(num_states_list):
        print("num states", num_states)
        for rep in range(num_reps):
            print(rep)
            posterior = generate_posterior_samples(num_samples, num_states)

        
            r_sa = np.mean(posterior, axis=1)
            #print("rsa", r_sa)
            init_distribution = np.ones(num_states)/num_states  #uniform distribution
            mdp_env = mdp.MachineReplacementMDP(num_states, r_sa, gamma, init_distribution)
            #run CVaR optimization, just the robust version since we don't have demos
            u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
            
            # print("solving for CVaR optimal policy")
            posterior_probs = np.ones(num_samples) / num_samples  #uniform dist since samples from MCMC
            import time
            t = time.time()
            cvar_opt_usa, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, posterior, posterior_probs, alpha, False, lamda)
            run_times[rep,i]  = time.time() - t


print(run_times)
print(np.mean(run_times, axis=0))
print(np.std(run_times, axis=0))
import os 
if not os.path.exists('./results/stress_test/'):
    os.makedirs('./results/stress_test/')
np.savetxt("./results/stress_test/machine_replace_states.csv", run_times, delimiter=",")