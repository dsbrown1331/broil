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


