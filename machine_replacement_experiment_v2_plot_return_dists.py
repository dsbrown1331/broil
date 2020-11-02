import bayesian_irl
import mdp_worlds
import utils
import mdp
import numpy as np
import scipy
import random
import generate_efficient_frontier
from machine_replacement_experimentv2 import generate_posterior_samples


if __name__=="__main__":
    seed = 1234
    np.random.seed(seed)
    scipy.random.seed(seed)
    random.seed(seed)
    num_states = 4
    num_samples = 200000
    #r_noop = np.array([0,0,-100])
    #r_repair = np.array([-50,-50,-50])
    gamma = 0.95
    alpha = 0.99
    
    lambdas_to_plot = [0.0,0.5,0.95]

    posterior = generate_posterior_samples(num_samples)

    lambda_usas = []
    lambdas = []
    #read in the u_sa's from the file
    f = open('./results/machine_replacement/policy_usas.txt')
    mean = False
    for line in f:
        print(line)
        if "--" in line:
            #parse header
            if "mean" in line:
                mean = True
                continue
            else:
                #get lambda value
                lambdas.append(float(line.strip().split(" ")[1]))
                mean = False
        
        else:
            items = line.strip().split(",")
            #parse line
            if mean:
                mean_usa = np.array([float(i) for i in items])
            else:
                #parse lambda
                lambda_usas.append(np.array([float(i) for i in items]))

    print(mean_usa)
    for l in lambda_usas:
        print(l)

    print()
    print(lambdas)


    #plot the mean returns versus the lambda =0 returns
    import matplotlib.pyplot as plt
    to_histogram = []
    label = []
    for l in lambdas_to_plot:
        usas = lambda_usas[lambdas.index(l)]
        to_histogram.append(np.dot(posterior.transpose(), usas))
        label.append("$\lambda={}$".format(l))

    to_histogram.append(np.dot(posterior.transpose(), mean_usa))
    label.append("mean ($\lambda=1.0$)")
    
    plt.hist(to_histogram, 100, label=label, stacked=False, fill=False, histtype='step', linewidth=2)
    plt.legend()
    plt.xlim(-2500, 0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.xlabel('Discounted return',fontsize=20)
    plt.ylabel('Number of runs',fontsize=20)
    plt.tight_layout()

    plt.show()