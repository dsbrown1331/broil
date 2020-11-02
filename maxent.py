import numpy as np
import numpy.random as rand
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm


n_actions = 4
        
def build_trans_mat_gridworld(num_rows, num_cols, terminals):
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9 
  # ...
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a 
  # special zero-reward terminal state (25) with no available actions
  num_states_plus = num_rows * num_cols + 1 #+1 for terminal sink with no reward
  trans_mat = np.zeros((num_states_plus,4,num_states_plus))
  
  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state, 
  # even itself, making it terminal, and state 24 is handled specially below.
  
  # Action 0 = left
  for s in range(num_rows * num_cols):
    if s%num_cols > 0:
      trans_mat[s,0,s-1] = 1
    else:
      trans_mat[s,0,s] = 1

  # Action 1 = right
  for s in range(num_rows * num_cols):
    if s%num_cols < num_cols - 1:
      trans_mat[s,1,s+1] = 1
    else:
      trans_mat[s,1,s] = 1

  
  # Action 2 = up
  for s in range(num_rows * num_cols):
    if s >= num_cols:
      trans_mat[s,2,s-num_cols] = 1
    else:
      trans_mat[s,2,s] = 1


  # Action 3 = down
  for s in range(num_rows * num_cols):
    if s < (num_rows - 1)*num_cols :
      trans_mat[s,3,s+num_cols] = 1
    else:
      trans_mat[s,3,s] = 1
  
  
      
  # Finally, goal state always goes to zero reward terminal state
  #first zero out all transitions
  for s in terminals:
    for a in range(4):
      for s2 in range(num_rows * num_cols):
        trans_mat[s,a,s2] = 0 #
    for a in range(4):
      trans_mat[s,a,num_rows * num_cols] = 1  #deterministic transition to sink with zero reward
        
  return trans_mat


           
def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, beta):
  """
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S
  
  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  #print "r_weights"
  #print r_weights
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  #recursively compute Z's
  Zs = np.zeros(n_states)
  Zs[n_states-1] = 1 #set sink Z_terminal = 1
  Za = np.zeros((n_states, n_actions))
  for t in range(horizon): #loop over horizon
    #print "horizon", t
    for s in range(n_states): #loop over states
      for a in range(n_actions):      #loop over actions
        e_reward = np.exp(beta*np.dot(r_weights,state_features[s])) #exp(reward(s|theta))
        Za[s,a] = np.sum([trans_mat[s,a,k] * e_reward * Zs[k] for k in range(n_states)])
        
    for s in range(n_states):
      Zs[s] = np.sum([Za[s,j] for j in range(n_actions)])
      if s == n_states - 1: #last state is hard coded as the sink
        Zs[s] += 1   
    #print "Z_aij"
    #print Za
    #print "Z_si"
  #print "Zs"
  #print Zs
    
  #local action probability computation
  policy = np.zeros((n_states,n_actions))
  for s in range(n_states):
    for a in range(n_actions):
      policy[s,a] = Za[s,a] / Zs[s]
  return policy, Za


  
def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s
  
  return: a size S array of expected state visitation frequencies
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  Dst = np.zeros((n_states, horizon))
  #recursively compute for t = 
  for t in range(horizon):
    if t is 0:
      Dst[:,0] = start_dist
    else:
      for k in range(n_states):
        for s in range(n_states):
          for a in range(n_actions):
            Dst[k,t] += Dst[s,t-1] * policy[s,a] * trans_mat[s,a,k]
  #sum frequencies          
  state_freq = np.sum(Dst, axis=1)
  print(state_freq)
  #for i in range(horizon):
  #  print "Dst sum", i
  #  print np.sum(Dst[:,i])
  #print "state_freq sum"
  #print np.sum(state_freq)
  return state_freq
  


def maxEntIRL(mdp_env, demos, seed_weights, max_epochs, horizon, learning_rate, beta=10, precision=0.00001, norm=None):
  """
  Compute a MaxEnt reward function from demonstration trajectories
  
  trans_mat: an S x A x S' array that describes transition probabilites from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size
  beta: inverse temperature parameter (I've found this algo is quite sensitive to it)
  precision: allows early stoppping by checking difference between expected feature counts to see if converged
  
  return: a size F array of reward weights
  """

  #compute transition probs
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld(mdp_env.num_rows, mdp_env.num_cols, mdp_env.terminals)
  num_features = mdp_env.get_reward_dimensionality()
  state_features = np.concatenate((mdp_env.state_features, np.zeros((1,num_features))), axis=0)#np.eye(26,25)  # Terminal state has no features, forcing zero reward

  #add terminal sink to start distribution
  start_dist = np.concatenate((mdp_env.init_dist,np.zeros(1)))

  #initialize weights
  r_weights = np.copy(seed_weights)
  
  #calculate expected feature counts
  n_features = np.shape(state_features)[1]
  f_count = np.zeros(n_features)
  for traj in demos:
    for state in traj:
        f_count += state_features[state]
        #print f_count
  f_count /= len(demos)
  #print f_count
  min_grad_norm = np.inf
  grad_norms = []
  for epoch in range(max_epochs):
    print("--------epoch", epoch, "---------")
    #test calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features)
    policy, Z_sa = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, beta)
    #print "policy"
    #print policy
    state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
    #print "state_freq"
    #print state_freq

    #compute gradient step (gradient ascent, since we're maximizing entropy!)
    #compute learners expected feature counts
    n_states = np.shape(trans_mat)[0]
    f_count_learner = np.zeros(n_features)
    for s in range(n_states):
      f_count_learner += state_freq[s] * state_features[s]
    #print "learner feature counts"
    #print f_count_learner
    grad = f_count - f_count_learner
    grad_norm = np.linalg.norm(grad)
    grad_norms.append(grad_norm)
    print("grad", grad)
    
    #print grad
    #print "grad norm"
    #print grad_norms
    r_weights += learning_rate * grad
    if norm is not None:
      if norm == "l2":
        #projected gradient descent on l2 ball
        r_weights /= np.linalg.norm(r_weights)
      elif norm == "inf":
        r_weights[r_weights>1] = 1
        r_weights[r_weights<-1] = -1
    print("r",r_weights)

    if len(grad_norms) > 2 and abs(grad_norms[-2] - grad_norms[-1]) < precision:
    #   min_grad_norm = grad_norm
    #   early_stop_cnt = 0
    #   print("updating min grad norm", min_grad_norm)
    # else:
    #   #increment early stopping count
    #   early_stop_cnt += 1
    #   #stop training if not making progress for patience iterations
    #   if early_stop_cnt > patience:
      print("stoping early")
    #     print(grad_norms)
      break
    #print "new weights"
    #print np.reshape(r_weights, (5,5))
    
  #print "empirical expected feature counts"
  #print f_count
  #print "learned policy expected feature counts"
  #print f_count_learner
  print(np.reshape(np.dot(state_features[:-1], r_weights), (mdp_env.num_rows,mdp_env.num_cols)))
  
  
  return r_weights, grad_norms, state_features, policy[:-1,:] #all but sink
  
 

def calc_max_ent_u_sa(mdp_env, demos, max_epochs=1000, horizon=None, learning_rate = 0.01):
  import mdp
  import utils
  seed_weights = np.zeros(mdp_env.get_reward_dimensionality())

  # Parameters
  if horizon is None:
    horizon = mdp_env.num_states

  # Main algorithm call
  r_weights, grads, state_features, maxent_pi = maxEntIRL(mdp_env, demos, seed_weights, max_epochs, horizon, learning_rate, norm="l2")

  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(mdp_env.num_states):
      reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (mdp_env.num_rows,mdp_env.num_cols))
  print("learned reward function")
  print(reward_fxn)

  u_s = mdp.get_policy_state_occupancy_frequencies(maxent_pi, mdp_env)
  u_sa = mdp.stoch_policy_to_usa(maxent_pi, mdp_env)
  utils.print_policy_from_occupancies(u_sa, mdp_env)
  utils.print_stochastic_policy_action_probs(u_sa, mdp_env)
  
  return u_sa, r_weights, maxent_pi

 
if __name__ == '__main__':
  
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = np.eye(26,25)  # Terminal state has no features, forcing zero reward
  demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]]
  seed_weights = np.zeros(25)
  
  # Parameters
  n_epochs = 100
  horizon = 10
  learning_rate = 0.5
  n_states = np.shape(trans_mat)[0]
  start_dist = np.zeros(n_states)
  start_dist[0] = 1 #start at state 0 
  
  # Main algorithm call
  r_weights, grads = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, start_dist)
  
  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))
  
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
  plt.show()
