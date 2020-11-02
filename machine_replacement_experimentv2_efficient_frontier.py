import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random
import generate_efficient_frontier

def generate_reward_sample():
    #costs for no-op are -N(0,10-4) except last state that is -N(100,800)
    r_noop = []
    locs = 1/2
    scales = [20, 40, 80,190]
    for i in range(4):
        r_noop.append(-np.random.gamma(locs, scales[i], 1)[0])
    r_noop = np.array(r_noop)
    #print(r_noop)
    
    #costs for repair are -N(130,1) for all but last state where it is -N(130,20)
    r_repair = -100 + -1 * np.random.randn(4)
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

def generate_posterior_samples(num_samples):
    print("samples")
    all_samples = []
    for i in range(num_samples):
        r_sample = generate_reward_sample()
        all_samples.append(r_sample)
        # r_string = ""
        # for r in r_sample:
        #     r_string += "{:.1f}\t".format(r)
        # print(r_string)

    print("mean of posterior from samples")
    print(np.mean(all_samples, axis=0))
    #print(generate_reward_mean(3))

    posterior = np.array(all_samples)

    return posterior.transpose()  #each column is a reward sample


if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    num_states = 4
    num_samples = 2000
    #r_noop = np.array([0,0,-100])
    #r_repair = np.array([-50,-50,-50])
    gamma = 0.95
    alpha = 0.99
    lamda = 0.9

    posterior = generate_posterior_samples(num_samples)

    #print(generate_reward_sample())

    r_sa = np.mean(posterior, axis=1)
    #print("rsa", r_sa)
    init_distribution = np.ones(num_states)/num_states  #uniform distribution
    mdp_env = mdp.MachineReplacementMDP(num_states, r_sa, gamma, init_distribution)
    #print(mdp_env.Ps)
    print("mean MDP reward", r_sa)

    u_sa = mdp.solve_mdp_lp(mdp_env, debug=True)
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

    
    
    #print(posterior)
    #print(posterior.shape)


    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    
    # print("solving for CVaR optimal policy")
    posterior_probs = np.ones(num_samples) / num_samples  #uniform dist since samples from MCMC
    # cvar_opt_usa, cvar, exp_ret = mdp.solve_max_cvar_policy(mdp_env, u_expert, posterior, posterior_probs, alpha, False, lamda)
    
    # print("mean policy from posterior")
    # utils.print_stochastic_policy_action_probs(u_sa, mdp_env)
    # print("MAP/Mean policy from posterior")
    # utils.print_policy_from_occupancies(u_sa, mdp_env) 
    # print("rewards")
    # print(mdp_env.r_sa)

    # print("CVaR policy")
    # utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env)
    # utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env)

    # cvar_reward, q = mdp.solve_minCVaR_reward(mdp_env, u_expert, posterior, posterior_probs, alpha)
    # print("cvar reward", cvar_reward)


    #generate efficient frontier
    lambda_range = [0.0, 0.3, 0.5, 0.75, 0.95,0.99, 1.0]

    #generate_efficient_frontier.calc_frontier(mdp_env, u_expert, posterior, posterior_probs, lambda_range, alpha, debug=False)
    alpha_range = [0.99]
    import matplotlib.pyplot as plt
    for alpha in alpha_range:
        cvar_rets = generate_efficient_frontier.calc_frontier(mdp_env, u_expert, posterior, posterior_probs, lambda_range, alpha, debug=False)
        
        cvar_rets_array = np.array(cvar_rets)
        print(cvar_rets_array)
        #input()
        plt.figure()
        #plt.title(r"$\alpha = {}$".format(alpha))
        plt.plot(cvar_rets_array[:,0], cvar_rets_array[:,1], '-o')
        #go through and label the points in the figure with the corresponding lambda values
        unique_pts_lambdas = []
        unique_pts = []
        for i,pt in enumerate(cvar_rets_array):
            unique = True
            for upt in unique_pts:
                if np.linalg.norm(upt - pt) < 0.00001:
                    unique = False
                    break
            if unique:
                unique_pts_lambdas.append((pt[0], pt[1], lambda_range[i]))
                unique_pts.append(np.array(pt))
        #calculate offset
        offsetx = (np.max(cvar_rets_array[:,0]) - np.min(cvar_rets_array[:,0]))/30
        offsety = (np.max(cvar_rets_array[:,1]) - np.min(cvar_rets_array[:,1]))/17
        #print(offsetx)
        #input()
        for i,pt in enumerate(unique_pts_lambdas):
            if i in [0,1,2,4]:
                plt.text(pt[0] - 6.2*offsetx, pt[1] , r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
            elif i in [3]:
                plt.text(pt[0] - 6.2*offsetx, pt[1] - 1.2*offsety , r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
            elif i in [5]:
                plt.text(pt[0] - 5.5*offsetx, pt[1] - 1.5*offsety, r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
            else:
                plt.text(pt[0]-offsetx, pt[1] - 1.5*offsety, r"$\lambda = {}$".format(str(pt[2])), fontsize=19,  fontweight='bold')
        #plt.axis([-1.43, -1.25, -1.05, -0.99])
        plt.xticks(fontsize=18) 
        plt.yticks(fontsize=18) 
        plt.xlabel("Robustness (CVaR)", fontsize=20)
        plt.ylabel("Expected Return", fontsize=20)
        
        #plt.legend(loc='best', fontsize=15)
        plt.tight_layout()
        #plt.savefig("./figs/alpha{}_lavaambiguous.png".format(alpha))
        #plt.show()

    print("Using hard coded values, comment out to recalculate for new lambdas!!!!!")
    plt.show()

    
