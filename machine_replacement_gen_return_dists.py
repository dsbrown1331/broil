import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random
import generate_efficient_frontier
from machine_replacement import generate_posterior_samples

if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    num_states = 4
    num_samples = 2000
    gamma = 0.95
    alpha = 0.99
    

    posterior = generate_posterior_samples(num_samples)


    r_sa = np.mean(posterior, axis=1)
    init_distribution = np.ones(num_states)/num_states  #uniform distribution
    mdp_env = mdp.MachineReplacementMDP(num_states, r_sa, gamma, init_distribution)

    print("mean MDP reward", r_sa)

    u_sa = mdp.solve_mdp_lp(mdp_env, debug=True)

    #write out to file
    f = open('./results/machine_replacement/policy_usas.txt', 'w')
    f.write("--mean policy\n")
    utils.write_line(u_sa, f)


    print("mean policy from posterior")
    utils.print_stochastic_policy_action_probs(u_sa, mdp_env)
    print("MAP/Mean policy from posterior")
    utils.print_policy_from_occupancies(u_sa, mdp_env) 
    print("rewards")
    print(mdp_env.r_sa)
    print("expected value = ", np.dot(u_sa, r_sa))
    stoch_pi = utils.get_optimal_policy_from_usa(u_sa, mdp_env)
    print("expected return", mdp.get_policy_expected_return(stoch_pi, mdp_env))
    print("values", mdp.get_state_values(u_sa, mdp_env))
    print('q-values', mdp.get_q_values(u_sa, mdp_env))

    
    
    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    
    posterior_probs = np.ones(num_samples) / num_samples  #uniform dist since samples from MCMC
    

    #generate efficient frontier
    lambda_range = [0.0,0.5, 0.95]

    
    for i,lamda in enumerate(lambda_range):
        print("lambda = ", lamda)
        cvar_opt_usa, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, posterior, posterior_probs, alpha, False, lamda)
        
        print('action probs')
        utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env)
        stoch_pi = utils.get_optimal_policy_from_usa(cvar_opt_usa, mdp_env)
        print(stoch_pi[:,1])
        f.write('--lambda {}\n'.format(lamda))
        utils.write_line(cvar_opt_usa, f)
        

