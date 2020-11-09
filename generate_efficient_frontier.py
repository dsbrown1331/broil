import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


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
        
        print("Policy for lambda={} and alpha={}".format(lamda, alpha))
        utils.print_policy_from_occupancies(cvar_opt_usa, mdp_env)
        print("stochastic policy")
        utils.print_stochastic_policy_action_probs(cvar_opt_usa, mdp_env)
        print("CVaR of policy = {}".format(cvar_value))
        print("Expected return of policy = {}".format(exp_ret))
        cvar_exprews.append((cvar_value, exp_ret))
    return cvar_exprews

if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    #mdp_env = mdp_worlds.two_state_chain()
    #demonstrations = [(1,0), (0,0)]

    # mdp_env = mdp_worlds.machine_teaching_toy_featurized()
    # demonstrations = [(2,3),(5,0),(4,0),(3,2)]


    mdp_env = mdp_worlds.lava_ambiguous_aaai18()
    u_sa = mdp.solve_mdp_lp(mdp_env)
    #generate demo from state 5 to terminal
    demonstrations = utils.rollout_from_usa(5, 10, u_sa, mdp_env)
    print(demonstrations)

    lambda_range = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    alpha_range = [0.75, 0.9, 0.95, 0.99] #for alpha zero they are all the sa
    larger_alpha_range =  [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] # for plotting how varies if lambda = 0
    debug = False

    beta = 50.0
    step_stdev = 0.2
    birl = bayesian_irl.BayesianIRL(mdp_env, beta, step_stdev, debug=False)

    num_samples = 2000
    burn = 50
    skip = 2
    map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, num_samples)
    print("map_weights", map_w)
    map_r = np.dot(mdp_env.state_features, map_w)
    utils.print_as_grid(map_r, mdp_env)
    #print("Map policy")
    #utils.print_policy_from_occupancies(map_u, mdp_env)

    # print("chain")
    # for r in r_chain:
    #     print(r)

    worst_index = np.argmin(r_chain[:,1])
    print(r_chain[worst_index])
    print(np.sum(r_chain[:,1] < -0.82), "out of ", len(r_chain))

    r_chain_burned = r_chain[burn::skip]
    # print("chain after burn and skip")
    # for r in r_chain_burned:
    #     print(r)
    #input()
    worst_index = np.argmin(r_chain_burned[:,1])
    print(r_chain_burned[worst_index])
    print(np.sum(r_chain_burned[:, 1]< -0.82), "out of", len(r_chain_burned))
    #input()

    #run CVaR optimization, maybe just the robust version for now
    u_expert = np.zeros(mdp_env.num_actions * mdp_env.num_states)
    n = r_chain_burned.shape[0]
    posterior_probs = np.ones(n) / n  #uniform dist since samples from MCMC
    
    
    for alpha in alpha_range:
        cvar_rets = calc_frontier(mdp_env, u_expert, r_chain_burned.transpose(), posterior_probs, lambda_range, alpha, debug)
        
        cvar_rets_array = np.array(cvar_rets)
        print(cvar_rets_array)
        #input()
        plt.figure()
        #plt.title(r"$\alpha = {}$".format(alpha))
        plt.plot(cvar_rets_array[:,0], cvar_rets_array[:,1], '-o')
        #go through and label the points in the figure with the corresponding lambda values
        unique_pts_lambdas = []
        unique_pts = []
        for i,pt in enumerate(cvar_rets):
            unique = True
            for upt in unique_pts:
                if np.linalg.norm(upt - pt) < 0.00001:
                    unique = False
                    break
            if unique:
                unique_pts_lambdas.append((pt[0], pt[1], lambda_range[i]))
                unique_pts.append(np.array(pt))
        #calculate offset
        offsetx = (np.max(cvar_rets_array[:,0]) - np.min(cvar_rets_array[:,0]))/40
        #print(offsetx)
        #input()
        for pt in unique_pts_lambdas:
            plt.text(pt[0] + offsetx, pt[1], r"$\lambda = {}$".format(str(pt[2])), fontsize=16,  fontweight='bold')
        plt.axis([-1.43, -1.25, -1.05, -0.99])
        plt.xticks(fontsize=15) 
        plt.yticks(fontsize=15) 
        plt.xlabel("Robustness (CVaR)", fontsize=18)
        plt.ylabel("Expected Return", fontsize=18)
        
        #plt.legend(loc='best', fontsize=15)
        plt.tight_layout()
        plt.savefig("./figs/alpha{}_lavaambiguous.png".format(alpha))
        #plt.show()

        
    plt.show()

    # plt.figure()
    # #plt.title("across alphas")
    # lambda_zero_vals = []
    
    # for alpha in larger_alpha_range:
    #     cvar_rets = calc_frontier(mdp_env, u_expert, r_chain_burned, posterior_probs, [0.0], alpha, debug)
    #     lambda_zero_vals.append(cvar_rets[0]) #first one corresponds to lambda = 0
    # lambda_zero_vals = np.array(lambda_zero_vals)
    # plt.plot(lambda_zero_vals[:,0], lambda_zero_vals[:,1], '-bo')
    # #calculate offset
    # offsetx = (np.max(cvar_rets_array[:,0]) - np.min(cvar_rets_array[:,0]))/40
    # offsety = (np.max(cvar_rets_array[:,1]) - np.min(cvar_rets_array[:,1]))/40
    # for i,pt in enumerate(lambda_zero_vals):
    #     plt.text(pt[0]+offsetx, pt[1]+offsety, r"$\alpha = {}$".format(larger_alpha_range[i]), fontsize=16,  fontweight='bold')
    # plt.xticks(fontsize=15) 
    # plt.yticks(fontsize=15) 
    # plt.xlabel("Robustness (CVaR)", fontsize=18)
    # plt.ylabel("Expected Return", fontsize=18)
    # #plt.legend(loc='best', fontsize=15)
    # plt.tight_layout()
    # plt.savefig("./figs/alpha_range_lambda0_lavaambiguous.png")

    # plt.show()