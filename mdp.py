# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:39:34 2020

@author: dsbrown
"""


import numpy as np
import utils
from scipy.optimize import linprog
from interface import implements, Interface
import sys

#acts as abstract class
class MDP(Interface):
    def get_num_actions(self):
        pass

    def get_reward_dimensionality(self):
        pass

    def set_reward_fn(self, new_reward):
        pass

    def get_transition_prob(self, s1,a,s2):
        pass

    def get_num_states(self):
        pass

    def get_readable_actions(self, action_num):
        pass

    def get_state_action_rewards(self):
        pass

    def uses_linear_approximation(self):
        pass

    def transform_to_R_sa(self, reward_weights):
        #mainly used for BIRL to take hypothesis reward and transform it
        #take in representation of reward weights and return vectorized version of R_sa
        #R_sa = [R(s0,a0), .., R(sn,a0), ...R(s0,am),..., R(sn,am)]
        pass

    def get_transition_prob_matrices(self):
        #return a list of transition matrices for each action a_0 through a_m
        pass

        


class ChainMDP(implements(MDP)):
    #basic MDP class that has two actions (left, right), no terminal states and is a chain mdp with deterministic transitions
    def __init__(self, num_states, r_sa, gamma, init_dist):
        self.num_actions = 2
        self.num_rows = 1
        self.num_cols = num_states
        self.num_states =  num_states
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = []
       
        self.r_sa = r_sa

        self.init_states = []
        for s in range(self.num_states):
            if self.init_dist[s] > 0:
                self.init_states.append(s)


        self.P_left = self.get_transitions(policy="left")
        #print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        #print("P_right\n",self.P_right)
        self.Ps = [self.P_left, self.P_right]

    def get_transition_prob_matrices(self):
        return self.Ps

    def get_num_actions(self):
        return self.num_actions

    def transform_to_R_sa(self, reward_weights):
        #Don't do anything, reward_weights should be r_sa 
        assert(len(reward_weights) == len(self.r_sa))
        return reward_weights

    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "<"
        elif action_num == 1:
            return ">"
        else:
            print("error, only two possible actions")
            sys.exit()

    def get_num_states(self):
        return self.num_states

    def get_reward_dimensionality(self):
        return len(self.r_sa)
    
    def uses_linear_approximation(self):
        return False

    def set_reward_fn(self, new_reward):
        self.r_sa = new_reward

    def get_state_action_rewards(self):
        return self.r_sa

    def get_transition_prob(self, s1,a,s2):
        return self.Ps[a][s1][s2]

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":  #action 0
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if c > 0:
                        P_pi[cnt, cnt - 1] = 1.0
                    else:
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "right":  #action 1
            #always transition one to right unless already at right border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if c < self.num_cols - 1:
                        #transition to next state to right
                        P_pi[cnt, cnt + 1] = 1.0
                    else:
                        #self transition
                        P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi
        


class MachineReplacementMDP(ChainMDP):
    #basic MDP class that has two actions (left, right), no terminal states and is a chain mdp with deterministic transitions
    def __init__(self, num_states, r_sa, gamma, init_dist):
        self.num_actions = 2
        self.num_rows = 1
        self.num_cols = num_states
        self.num_states =  num_states
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = []
       
        self.r_sa = r_sa


        self.P_noop = self.get_transitions(policy="noop")
        #print("P_left\n",self.P_left)
        self.P_repair = self.get_transitions(policy="repair")
        #print("P_right\n",self.P_right)
        self.Ps = [self.P_noop, self.P_repair]


    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "noop" #no-op
        elif action_num == 1:
            return "repair" #repair
        else:
            print("error, only two possible actions")
            sys.exit()

    
    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "noop":  #action 0
            #always transition to one state farther in chain unless at the last state where you go to the beginning
            for c in range(self.num_cols):
                if c < self.num_cols - 1:
                    #continue to the right
                    P_pi[c, c + 1] = 1.0
                else:
                    #go back to the beginning
                    P_pi[c,0] = 1.0
            
        elif policy == "repair":  #action 1
            #always transition back to the first state
            for c in range(self.num_cols):
                P_pi[c,0] = 1.0
                
        return P_pi
        


class BasicGridMDP(implements(MDP)):
    #basic MDP class that has four actions, possible terminal states and is a grid with deterministic transitions
    def __init__(self, num_rows, num_cols, r_s, gamma, init_dist, terminals = [], debug=False):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = terminals
        self.debug = debug
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.r_s)
        #print("transformed R(s,a)", self.r_sa)

        self.init_states = []
        for s in range(self.num_states):
            if self.init_dist[s] > 0:
                self.init_states.append(s)


        self.P_left = self.get_transitions(policy="left")
        if self.debug: print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        if self.debug: print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        if self.debug: print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        if self.debug: print("P_down\n",self.P_down)
        self.Ps = [self.P_left, self.P_right, self.P_up, self.P_down] #actions:0,1,2,3


    def get_transition_prob_matrices(self):
        return self.Ps

    def get_num_actions(self):
        return self.num_actions

    def get_num_states(self):
        return self.num_states

   
    def get_readable_actions(self, action_num):
        if action_num == 0:
            return "<"
        elif action_num == 1:
            return ">"
        elif action_num == 2:
            return "^"
        elif action_num == 3:
            return "v"
        else:
            print("error, only four possible actions")
            sys.exit()


    def get_transition_prob(self, s1,a,s2):
        return self.Ps[a][s1][s2]

    #Note that I'm using r_s as the reward dim not r_sa!
    def get_reward_dimensionality(self):
        return len(self.r_s)

    #NOTE: the dimensionality still needs to be checked.
    def uses_linear_approximation(self):
        return False

    def get_state_action_rewards(self):
        return self.r_sa

    #assume new reward is of the form r_s
    def set_reward_fn(self, new_reward):
        self.r_s = new_reward
        #also update r_sa
        self.r_sa = self.transform_to_R_sa(self.r_s)



    #transform R(s) into R(s,a) for use in LP
    def transform_to_R_sa(self, reward_weights):
        #assume that reward_weights is r_s
        #tile to get r_sa from r

        '''input: numpy array R_s, output R_sa'''
        #print(len(R_s))
        #print(self.num_states)
        #just repeat values since R_sa = [R(s1,a1), R(s2,a1),...,R(sn,a1), R(s1,a2), R(s2,a2),..., R(sn,am)]
        assert(len(reward_weights) == self.num_states)
        return np.tile(reward_weights, self.num_actions)

    def get_transitions(self, policy):
        P_pi = np.zeros((self.num_states, self.num_states))
        if policy == "left":  #action 0 
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if c > 0:
                            P_pi[cnt, cnt - 1] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "right":  #action 1
            #always transition one to right unless already at right border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if c < self.num_cols - 1:
                            #transition to next state to right
                            P_pi[cnt, cnt + 1] = 1.0
                        else:
                            #self transition
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "up": #action 2
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if r > 0:
                            P_pi[cnt, cnt - self.num_cols] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        elif policy == "down":  #action 3
            #always transition one to left unless already at left border
            cnt = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    if cnt not in self.terminals: #no transitions out of terminal
                        if r < self.num_rows - 1:
                            P_pi[cnt, cnt + self.num_cols] = 1.0
                        else:
                            P_pi[cnt,cnt] = 1.0
                    #increment state count
                    cnt += 1
        return P_pi


class FeaturizedGridMDP(BasicGridMDP):


    def __init__(self,num_rows, num_cols, state_feature_matrix, feature_weights, gamma, init_dist, terminals = [], debug=False):
        self.num_actions = 4
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_states =  num_rows * num_cols
        self.gamma = gamma
        self.init_dist = init_dist
        self.terminals = terminals
        self.debug = debug

        self.init_states = []
        for s in range(self.num_states):
            if self.init_dist[s] > 0:
                self.init_states.append(s)

        
        self.P_left = self.get_transitions(policy="left")
        if self.debug: print("P_left\n",self.P_left)
        self.P_right = self.get_transitions(policy="right")
        if self.debug: print("P_right\n",self.P_right)
        self.P_up = self.get_transitions(policy="up")
        if self.debug: print("_up\n",self.P_up)
        self.P_down = self.get_transitions(policy="down")
        if self.debug: print("P_down\n",self.P_down)
        self.Ps = [self.P_left, self.P_right, self.P_up, self.P_down] #actions:0,1,2,3

        #figure out reward function
        self.state_features = state_feature_matrix
        self.feature_weights = feature_weights
        r_s = np.dot(self.state_features, self.feature_weights)
        #print("r_s", r_s)
        self.r_s = r_s
        self.r_sa = self.transform_to_R_sa(self.feature_weights)
        #print("transformed R(s,a)", self.r_sa)


    def get_reward_dimensionality(self):
        return len(self.feature_weights)

    def uses_linear_approximation(self):
        return True

    def set_reward_fn(self, new_reward):
        #input is the new_reward weights
        assert(len(new_reward) == len(self.feature_weights))
        #update feature weights
        self.feature_weights = new_reward.copy()
        #update r_s
        self.r_s = np.dot(self.state_features, new_reward)
        #update r_sa
        self.r_sa = np.tile(self.r_s, self.num_actions)


    def transform_to_R_sa(self, reward_weights):
        #assumes that inputs are the reward feature weights or state rewards
        #returns the vectorized R_sa 
        
        #first get R_s
        if len(reward_weights) == self.get_reward_dimensionality():
            R_s = np.dot(self.state_features, reward_weights)
        elif len(reward_weights) == self.num_states:
            R_s = reward_weights
        else:
            print("Error, reward weights should be features or state rewards")
            sys.exit()
        return np.tile(R_s, self.num_actions)



def get_windy_down_grid_transitions(mdp_env, slip_prob):
    num_rows, num_cols = mdp_env.num_rows, mdp_env.num_cols
    num_states = mdp_env.num_states
    terminals = mdp_env.terminals
    #action 0 LEFT
    P_left = np.zeros((num_states, num_states))
    #always transition one to left unless already at left border
    prob_slip = slip_prob
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                #check columns
                if c == 0:
                    P_left[cnt,cnt] += 1.0 - prob_slip #self loop if 
                    if r < num_rows - 1: # check if above last row
                        P_left[cnt, cnt + num_cols] += prob_slip
                    else:
                        P_left[cnt,cnt] += prob_slip
                else: #c > 0
                    P_left[cnt, cnt - 1] = 1.0 - prob_slip
                    if r < num_rows - 1: # check if above last row
                        P_left[cnt, cnt - 1 + num_cols] += prob_slip
                    else:
                        P_left[cnt,cnt - 1] += prob_slip
                    
            #increment state count
            cnt += 1

    #action 1 RIGHT
    P_right = np.zeros((num_states, num_states))
    #always transition one to right unless already at right border
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                if c < num_cols - 1:
                    #transition to next state to right
                    P_right[cnt, cnt + 1] += 1.0 - prob_slip
                    if r < num_rows - 1: # check if above last row
                        P_right[cnt, cnt + 1 + num_cols] += prob_slip
                    else:
                        P_right[cnt,cnt + 1] += prob_slip
                else: # c == num_cols - 1 (at right edge of world)
                    #self transition
                    P_right[cnt,cnt] = 1.0 - prob_slip
                    if r < num_rows - 1: # check if above last row
                        P_right[cnt, cnt + num_cols] += prob_slip
                    else: # bottom right corner
                        P_right[cnt,cnt] += prob_slip
            #increment state count
            cnt += 1
    #action 2 UP
    #Let's say it pushes you left or right with prob_slip / 2
    P_up = np.zeros((num_states, num_states))
    #always transition one to left unless already at left border
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                if r > 0:
                    P_up[cnt, cnt - num_cols] = 1.0 - prob_slip
                    if c == 0:
                        P_up[cnt, cnt - num_cols] += prob_slip / 2 #go up left and run into wall
                        P_up[cnt, cnt - num_cols + 1] += prob_slip / 2
                    elif c == num_cols - 1:
                        P_up[cnt, cnt - num_cols - 1] += prob_slip / 2 #go up left and run into wall
                        P_up[cnt, cnt - num_cols] += prob_slip / 2
                    else:
                        P_up[cnt, cnt - num_cols - 1] += prob_slip / 2 #go up left 
                        P_up[cnt, cnt - num_cols + 1] += prob_slip / 2 #go up and right
                else: #r == 0
                    P_up[cnt, cnt] = 1.0 - prob_slip
                    if c == 0:
                        P_up[cnt, cnt] += prob_slip / 2 #go up left and run into wall
                        P_up[cnt, cnt + 1] += prob_slip / 2
                    elif c == num_cols - 1:
                        P_up[cnt, cnt] += prob_slip / 2 #go up left and run into wall
                        P_up[cnt, cnt - 1] += prob_slip / 2
                    else:
                        P_up[cnt, cnt - num_cols - 1] += prob_slip / 2 #go up left 
                        P_up[cnt, cnt - num_cols + 1] += prob_slip / 2 #go up and right
            #increment state count
            cnt += 1
    #action 3 DOWN
    P_down = np.zeros((num_states, num_states))
    #always transition one to left unless already at left border
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                if r < num_rows - 1:
                    P_down[cnt, cnt + num_cols] = 1.0 - prob_slip
                    if c == 0:
                        P_down[cnt, cnt + num_cols] += prob_slip / 2 #go up left and run into wall
                        P_down[cnt, cnt + num_cols + 1] += prob_slip / 2
                    elif c == num_cols - 1:
                        P_down[cnt, cnt + num_cols - 1] += prob_slip / 2 #go up left and run into wall
                        P_down[cnt, cnt + num_cols] += prob_slip / 2
                    else:
                        P_down[cnt, cnt + num_cols - 1] += prob_slip / 2 #go up left 
                        P_down[cnt, cnt + num_cols + 1] += prob_slip / 2 #go up and right
                else: #r == num_rows - 1
                    P_down[cnt, cnt] = 1.0 - prob_slip
                    if c == 0:
                        P_down[cnt, cnt] += prob_slip / 2 #go up left and run into wall
                        P_down[cnt, cnt + 1] += prob_slip / 2
                    elif c == num_cols - 1:
                        P_down[cnt, cnt] += prob_slip / 2 #go up left and run into wall
                        P_down[cnt, cnt - 1] += prob_slip / 2
                    else:
                        P_down[cnt, cnt - 1] += prob_slip / 2 #go up left 
                        P_down[cnt, cnt + 1] += prob_slip / 2 #go up and right
            #increment state count
            cnt += 1
    Ps = [P_left, P_right, P_up, P_down] #actions:0,1,2,3
    return Ps


#just assume that every action away from left and right edges has windy prob of pushing down when taking any action
def get_windy_down_const_prob_transitions(mdp_env, slip_prob):
    num_rows, num_cols = mdp_env.num_rows, mdp_env.num_cols
    num_states = mdp_env.num_states
    terminals = mdp_env.terminals
    #action 0 LEFT
    P_left = np.zeros((num_states, num_states))
    #always transition one to left unless already at left border
    prob_slip = slip_prob
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                #check columns
                if c == 0:
                    P_left[cnt,cnt] = 1.0
                elif c == num_cols - 1:
                    P_left[cnt,cnt - 1] = 1.0

                else: #c > 0 and c < num_cols - 1 so possibly slip
                    P_left[cnt, cnt - 1] = 1.0 - prob_slip
                    if r < num_rows - 1: # check if above last row
                        P_left[cnt, cnt + num_cols] += prob_slip #slip down
                    else:
                        P_left[cnt,cnt] += prob_slip
                    
            #increment state count
            cnt += 1

    #action 1 RIGHT
    P_right = np.zeros((num_states, num_states))
    #always transition one to right unless already at right border
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                if c == 0:
                    P_right[cnt,cnt+1] = 1.0

                elif c < num_cols - 1:
                    #transition to next state to right or go down instead
                    P_right[cnt, cnt + 1] += 1.0 - prob_slip
                    if r < num_rows - 1: # check if above last row
                        P_right[cnt, cnt + num_cols] += prob_slip
                    else:
                        P_right[cnt,cnt] += prob_slip
                
                else: # c == num_cols - 1 (at right edge of world)
                    #self transition
                    P_right[cnt,cnt] = 1.0 #just bump into wall on right
            #increment state count
            cnt += 1
    #action 2 UP
    #Let's say it pushes you left or right with prob_slip / 2
    P_up = np.zeros((num_states, num_states))
    #always transition one to left unless already at left border
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                # if cnt == 7:
                   # print("hey")
                if r > 0:
                    P_up[cnt, cnt - num_cols] = 1.0 - prob_slip
                    if c == 0:
                        P_up[cnt, cnt - num_cols] += prob_slip #go up
                    elif c == num_cols - 1:
                        P_up[cnt, cnt - num_cols] += prob_slip #go up
                    else: #maybe go down instead
                        if r < num_rows - 1:
                            P_up[cnt, cnt + num_cols] += prob_slip 
                        else:
                            P_up[cnt, cnt] += prob_slip 
                else: #r == 0
                    P_up[cnt, cnt] = 1.0 - prob_slip
                    if c == 0: # no slip
                        P_up[cnt, cnt] += prob_slip 
                    elif c == num_cols - 1: #no slip
                        P_up[cnt, cnt] += prob_slip 
                    else: #slip down maybe
                        P_up[cnt, cnt + num_cols] += prob_slip 
            #increment state count
            cnt += 1
    #action 3 DOWN
    P_down = np.zeros((num_states, num_states))
    #go left or right or down when going down unless not in the wind
    cnt = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if cnt not in terminals: #no transitions out of terminal
                if r < num_rows - 1:
                    P_down[cnt, cnt + num_cols] = 1.0 - prob_slip
                    if c == 0: #no wind
                        P_down[cnt, cnt + num_cols] += prob_slip 
                    elif c == num_cols - 1: # no wind
                        P_down[cnt, cnt + num_cols] += prob_slip 
                    else:
                        P_down[cnt, cnt + num_cols - 1] += prob_slip / 2 #go down left 
                        P_down[cnt, cnt + num_cols + 1] += prob_slip / 2 #go down and right
                else: #r == num_rows - 1
                    P_down[cnt, cnt] = 1.0 #just go down
            cnt += 1
    Ps = [P_left, P_right, P_up, P_down] #actions:0,1,2,3
    return Ps




def get_state_values(occupancy_frequencies, mdp_env):
    num_states, gamma = mdp_env.num_states, mdp_env.gamma
    r_sa = mdp_env.get_state_action_rewards()
    #get optimal stochastic policy
    stochastic_policy = utils.get_optimal_policy_from_usa(occupancy_frequencies, mdp_env)
    
    reward_policy = get_policy_rewards(stochastic_policy, r_sa)
    transitions_policy = get_policy_transitions(stochastic_policy, mdp_env)
    A = np.eye(num_states) - gamma * transitions_policy 
    b = reward_policy
    #solve for value function
    state_values = np.linalg.solve(A, b)

    return state_values
    

def get_q_values(occupancy_frequencies, mdp_env):
    num_actions, gamma = mdp_env.num_actions, mdp_env.gamma
    r_sa = mdp_env.get_state_action_rewards()
    #get state values
    state_values = get_state_values(occupancy_frequencies, mdp_env)
    #get state-action values
    Ps = tuple(mdp_env.Ps[i] for i in range(num_actions))
    P_column = np.concatenate(Ps, axis=0)
    #print(P_column)
    q_values = r_sa + gamma * np.dot(P_column, state_values)
    return q_values
    

# def get_q_values(occupancy_frequencies, mdp_env):
#     num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
#     stochastic_policy = utils.get_optimal_policy_from_usa(occupancy_frequencies, mdp_env)
#     reward_policy = get_policy_rewards(stochastic_policy, reward_sa)
#     transitions_policy = get_policy_transitions(stochastic_policy, mdp_env)
#     A = np.eye(num_states) - gamma * transitions_policy 
#     b = reward_policy
    
#     state_values = np.linalg.solve(A, b)
#     Ps = tuple(self.mdp_env.Ps[i] for i in range(num_actions))
#     P_column = np.concatenate(Ps, axis=0)
#     #print(P_column)
#     q_values = reward_sa + gamma * np.dot(P_column, state_values)


def solve_mdp_lp(mdp_env, reward_sa=None, debug=False):
    '''method that uses Linear programming to solve MDP
        if reward_sa is not None, then it uses reward_sa in place of mdp_env.r_sa
    
    '''

    I_s = np.eye(mdp_env.num_states)
    gamma = mdp_env.gamma

    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)

    # if mdp_env.num_actions == 4:
    #     A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
    #                     I_s - gamma * mdp_env.P_right.transpose(),
    #                     I_s - gamma * mdp_env.P_up.transpose(),
    #                     I_s - gamma * mdp_env.P_down.transpose()),axis =1)
    # else:
        
    #     A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
    #                     I_s - gamma * mdp_env.P_right.transpose()),axis =1)
    b_eq = mdp_env.init_dist
    if reward_sa is not None:
        c = -1.0 * reward_sa  #we want to maximize r_sa^T c so make it negative since scipy minimizes by default
    else:
        c = -1.0 * mdp_env.r_sa  #we want to maximize r_sa^T c so make it negative since scipy minimizes by default

    sol = linprog(c, A_eq=A_eq, b_eq = b_eq)
    #minimize:
    #c @ x
    #such that:
    #A_ub @ x <= b_ub
    #A_eq @ x == b_eq
    #all variables are non-negative by default
    #print(sol)

    if debug: print("expeced value MDP LP", -sol['fun'])  #need to negate the value to get the maximum
    #print("state_action occupancies", sol['x'])
    u_sa = sol['x'] 

    #print("expected value dot product", np.dot(u_sa, mdp_env.r_sa))
    #calculate the optimal policy
    return u_sa


def solve_lpal_policy(mdp_env, u_expert, debug=False):
    '''input mdp_env: the mdp
        u_expert: the expected feature coutns of the expert
            #using the LP formulation that Marek derived that works for both positive and negative weights
        returns: u_sa from the LPAL algorithm
    '''

    w_dim = len(u_expert)

    #decision variable are [u_sa, B]
    n_states = mdp_env.num_states
    n_actions = mdp_env.num_actions
    Phi = mdp_env.state_features

    k = mdp_env.num_states * mdp_env.num_actions

    I_s = np.eye(mdp_env.num_states)
    gamma = mdp_env.gamma

    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)
    A_eq = np.concatenate((A_eq, np.zeros((n_states,1))), axis=1) #add zeros for the B variable

    b_eq = mdp_env.init_dist

    #LPAL maxmin constraints
    #first create a feature matrix that works for all actions
    Phi_LPAL = np.concatenate([Phi for _ in range(n_actions)] , axis=0)

    A_leq_maxmin1 = np.concatenate((-Phi_LPAL.transpose(), np.ones((w_dim,1))), axis=1)
    A_leq_maxmin2 = np.concatenate((Phi_LPAL.transpose(), -np.ones((w_dim,1))), axis=1)
    A_leq_maxmin = np.concatenate((A_leq_maxmin1, A_leq_maxmin2), axis=0)
    b_geq_maxmin = np.concatenate((-u_expert, u_expert), axis=0)

    #constraints for u_sa >=0 
    A_u_geq0 = -np.eye(k, M=k+1)  #negative since constraint needs to be Ax<=b
    b_u_geq0 = np.zeros(k)

    #stack all the constraints on top of each other
    A_leq = np.concatenate((A_leq_maxmin, A_u_geq0), axis=0)
    b_geq = np.concatenate((b_geq_maxmin, b_u_geq0), axis=0)

 
    #decision variables are [u_sa, B]
    c = np.concatenate((np.zeros(k), np.ones(1)))  #we want to minimize B 

    sol = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_leq, b_ub=b_geq, bounds=(None, None))
    #minimize:
    #c @ x
    #such that:
    #A_ub @ x <= b_ub
    #A_eq @ x == b_eq
    #all variables are non-negative by default
    #print(sol)

    if debug: print("expeced value MDP LP", -sol['fun'])  #need to negate the value to get the maximum
    #print("state_action occupancies", sol['x'])
    u_sa = sol['x'][:k]
    B = sol['x'][k] 

    #print("expected value dot product", np.dot(u_sa, mdp_env.r_sa))
    #calculate the optimal policy
    return u_sa


def solve_lpal_policy_old(mdp_env, u_expert, debug=False):
    '''input mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
            
        returns: u_sa from the LPAL algorithm
    '''
    n_states = mdp_env.num_states
    n_actions = mdp_env.num_actions
    Phi = mdp_env.state_feature_matrix

    k = mdp_env.num_states * mdp_env.num_actions

    I_s = np.eye(mdp_env.num_states)
    gamma = mdp_env.gamma

    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)
    A_eq = np.concatenate((A_eq, np.zeros(n_states,1))) #add zeros for the B variable

    b_eq = mdp_env.init_dist

    #LPAL maxmin constraints
    pos_neg_features = np.concatenate((Phi, -Phi), axis=1)
    Phi_LPAL = np.concatenate([pos_neg_features for _ in range(n_actions)] , axis=0)

    A_leq_maxmin = np.concatenate((np.ones(2*k,1), -Phi_LPAL.transpose()), axis=1)
    b_geq_maxmin = -np.concatenate((u_expert, -u_expert), axis=0)

    #constraints for u_sa >=0 
    A_u_geq0 = -np.eye(k, M=k+1)  #negative since constraint needs to be Ax<=b
    b_u_geq0 = np.zeros(k)

    #stack all the constraints on top of each other
    A_leq = np.concatenate((A_leq_maxmin, A_u_geq0), axis=0)
    b_geq = np.concatenate((b_geq_maxmin, b_u_geq0), axis=0)

 
    #decision variables are [u_sa, B]
    c = -np.concatenate((np.zeros(k), np.ones(1)))  #we want to maximize B c so make it negative since scipy minimizes by default

    sol = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_leq, b_up=b_geq, bounds=(None, None))
    #minimize:
    #c @ x
    #such that:
    #A_ub @ x <= b_ub
    #A_eq @ x == b_eq
    #all variables are non-negative by default
    #print(sol)

    if debug: print("expeced value MDP LP", -sol['fun'])  #need to negate the value to get the maximum
    #print("state_action occupancies", sol['x'])
    u_sa = sol['x'][:k]
    B = sol['x'][k] 

    #print("expected value dot product", np.dot(u_sa, mdp_env.r_sa))
    #calculate the optimal policy
    return u_sa

def solve_cvar_expret_fixed_policy(mdp_env, u_policy, u_expert, posterior_rewards, p_R, alpha, debug=False):
    '''
    Solves for CVaR and expectation with respect to BROIL baseline regret formulation using u_expert as the baseline
    input 
        mdp_env: the mdp
        u_policy: the pre-optimized policy
        u_expert: the state-action occupancies of the expert
        posterior_rewards: a matrix with each column a reward hypothesis or each column a weight vector
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case
        lamda: the amount to weight expected return versus CVaR if 0 then fully robust, if 1, then fully return maximizing

        returns: tuple (u_cvar, cvar) the occupancy frequencies of the policy optimal wrt cvar and the actual cvar value
    '''


    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim! This isn't the k dim of the weights!
    k = mdp_env.num_states * mdp_env.num_actions

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()

    posterior_probs = p_R
    #new objective is 
    #max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

    #so the decision variables are (in the following order) sigma, and all the z's from the ReLUs

    #we want to maximize so take the negative of this vector and minimize via scipy 
    c_cvar = -1. * np.concatenate((np.ones(1),                 #for sigma
                        -1.0/(1.0 - alpha) * posterior_probs))  #for the auxiliary variables z

    #constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

    #create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
    # and n rows (one for each z variable)
    auxiliary_constraints = np.concatenate((np.ones((n,1)), -np.eye(n)),axis=1)
    
    #add the upper bounds for these constraints:
    #check to see if we have mu or u
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        print(R.shape)
        print(u_policy.shape)
        print(posterior_rewards.transpose().shape)
        print(u_expert.shape)

        auxiliary_b = np.dot(R.transpose(), u_policy) - np.dot(posterior_rewards.transpose(), u_expert)
    else:
        #no feature approximation for reward, just tabular
        auxiliary_b = np.dot(R.transpose(), u_policy - u_expert)

    #add the non-negativitity constraints for z(R). 
    auxiliary_z_geq0 = np.concatenate((np.zeros((n,1)), -np.eye(n)), axis=1)
    auxiliary_bz_geq0 = np.zeros(n)

    
    A_cvar = np.concatenate((auxiliary_constraints,
                            auxiliary_z_geq0), axis=0)
    b_cvar = np.concatenate((auxiliary_b, auxiliary_bz_geq0))

    #solve the LP
    sol = linprog(c_cvar, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    if debug: print("solution to optimizing CVaR")
    if debug: print(sol)
    
    if sol['success'] is False:
        #print(sol)
        print("didn't solve correctly!")
        input("Continue?")
    #the solution of the LP corresponds to the CVaR
    var_sigma = sol['x'][0] #get sigma (this is VaR (at least close))
    cvar = -sol['fun'] #get the cvar of the input policy (negative since we minimized negative objective)
    
    #calculate expected return of optimized policy
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        expected_perf_expert = np.dot(posterior_probs, np.dot(posterior_rewards.transpose(), u_expert))
    else:
        expected_perf_expert = np.dot( np.dot(R, posterior_probs), u_expert)
    cvar_exp_ret = np.dot( np.dot(R, posterior_probs), u_policy) - expected_perf_expert

    if debug: 
        print("CVaR = ", cvar)
        print("Expected return = ", cvar_exp_ret)
    
    
    return cvar, cvar_exp_ret



def solve_max_cvar_policy(mdp_env, u_expert, posterior_rewards, p_R, alpha, debug=False, lamda = 0.0):
    '''input mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
        posterior_rewards: a matrix with each column a reward hypothesis or each column a weight vector
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case
        lamda: the amount to weight expected return versus CVaR if 0 then fully robust, if 1, then fully return maximizing

        returns: tuple (u_cvar, cvar) the occupancy frequencies of the policy optimal wrt cvar and the actual cvar value
    '''


    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim!
    k = mdp_env.num_states * mdp_env.num_actions

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()

    posterior_probs = p_R
    #new objective is 
    #max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

    #so the decision variables are (in the following order) all the u(s,a) and sigma, and all the z's.

    #we want to maximize so take the negative of this vector and minimize via scipy 
    u_coeff = np.dot(R, posterior_probs)
    c_cvar = -1. * np.concatenate((lamda * u_coeff, #for the u(s,a)'s (if lamda = 0 then no in objective, this is the lambda * p^T R^T u)
                        (1-lamda) * np.ones(1),                 #for sigma
                        (1-lamda) * -1.0/(1.0 - alpha) * posterior_probs))  #for the auxiliary variables z

    #constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

    #create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
    # and n rows (one for each z variable)
    auxiliary_constraints = np.zeros((n, k + 1 + n))
    for i in range(n):
        z_part = np.zeros(n)
        z_part[i] = -1.0 #make the part for the auxiliary variable >= the part in the relu
        z_row = np.concatenate((-R[:,i],  #-R_i(s,a)'s
                                np.ones(1),    #sigma
                                z_part))
        auxiliary_constraints[i,:] = z_row

    #add the upper bounds for these constraints:
    #check to see if we have mu or u
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        auxiliary_b = -1. * np.dot(posterior_rewards.transpose(), u_expert)
    else:
        auxiliary_b = -1. * np.dot(R.transpose(), u_expert)

    #add the non-negativitity constraints for the vars u(s,a) and z(R). 
    #mu's greater than or equal to zero
    auxiliary_u_geq0 = -np.eye(k, M=k+1+n)  #negative since constraint needs to be Ax<=b
    auxiliary_bu_geq0 = np.zeros(k)

    auxiliary_z_geq0 = np.concatenate((np.zeros((n, k+1)), -np.eye(n)), axis=1)
    auxiliary_bz_geq0 = np.zeros(n)

    #don't forget the normal MDP constraints over the mu(s,a) terms
    I_s = np.eye(num_states)
    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)

    # if mdp_env.num_actions == 4:
    #     A_eq = np.concatenate(( I_s - gamma * mdp_env.P_left.transpose(),
    #                             I_s - gamma * mdp_env.P_right.transpose(),
    #                             I_s - gamma * mdp_env.P_up.transpose(),
    #                             I_s - gamma * mdp_env.P_down.transpose()),axis =1)
    # else:
    #     A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
    #                            I_s - gamma * mdp_env.P_right.transpose()),axis =1)
    b_eq = mdp_env.init_dist
    A_eq_plus = np.concatenate((A_eq, np.zeros((mdp_env.num_states,1+n))), axis=1)  #add zeros for sigma and the auxiliary z's

    A_cvar = np.concatenate((auxiliary_constraints,
                            auxiliary_u_geq0,
                            auxiliary_z_geq0), axis=0)
    b_cvar = np.concatenate((auxiliary_b, auxiliary_bu_geq0, auxiliary_bz_geq0))

    #solve the LP
    sol = linprog(c_cvar, A_eq=A_eq_plus, b_eq = b_eq, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    if debug: print("solution to optimizing CVaR")
    if debug: print(sol)
    
    if sol['success'] is False:
        #print(sol)
        print("didn't solve correctly!")
        input("Continue?")
    #the solution of the LP corresponds to the CVaR
    var_sigma = sol['x'][k] #get sigma (this is VaR (at least close))
    cvar_opt_usa = sol['x'][:k]

    #calculate the CVaR of the solution
    if k != len(u_expert):
        relu_part = var_sigma * np.ones(n) - np.dot(np.transpose(R), cvar_opt_usa) + np.dot(np.transpose(posterior_rewards), u_expert)
    else:
        relu_part = var_sigma * np.ones(n) - np.dot(np.transpose(R), cvar_opt_usa) + np.dot(np.transpose(R), u_expert)
    #take max with zero
    relu_part[relu_part < 0] = 0.0
    cvar = var_sigma - 1.0/(1 - alpha) * np.dot(posterior_probs, relu_part)

    #calculate expected return of optimized policy
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        exp_baseline_perf = np.dot(posterior_probs, np.dot(posterior_rewards.transpose(), u_expert))
    
    else:
        exp_baseline_perf = np.dot(np.dot(R, posterior_probs), u_expert)


    cvar_exp_ret = np.dot( np.dot(R, posterior_probs), cvar_opt_usa) - exp_baseline_perf

    if debug: print("CVaR = ", cvar)
    if debug: print("policy u(s,a) = ", cvar_opt_usa)
    cvar_opt_stoch_pi = utils.get_optimal_policy_from_usa(cvar_opt_usa, mdp_env)
    if debug: print("CVaR opt stochastic policy")
    if debug: print(cvar_opt_stoch_pi)

    if debug:
        if k != len(u_expert):
            policy_losses = np.dot(R.transpose(), cvar_opt_usa)  - np.dot(posterior_rewards.transpose(), u_expert)
        else:
            policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)
        print("policy losses:", policy_losses)
    if debug: 
        if k != len(u_expert):
            print("expert returns:", np.dot(posterior_rewards.transpose(), u_expert))
        else:
            print("expert returns:", np.dot(R.transpose(), u_expert))
    if debug: print("my returns:", np.dot(R.transpose(), cvar_opt_usa))

    return cvar_opt_usa, cvar, cvar_exp_ret


def solve_minCVaR_reward(mdp_env, u_expert, posterior_rewards, p_R, alpha):
    '''
    Solves the dual problem
      input:
        mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
        R: a matrix with each column a reward hypothesis
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case

       output:
        The adversarial reward and the q weights on the reward posterior. Optimizing for this reward should yield the CVaR optimal policy

    '''
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim!
    k = mdp_env.num_states * mdp_env.num_actions

    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    p0 = mdp_env.init_dist

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()


    #k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    posterior_probs = p_R
    #objective is min p_0^Tv - u_E^T R q

    #the decision variables are (in the following order) q (an element for each reward in the posterior) and v(s) for all s
    #coefficients on objective
    if k != len(u_expert):
        #function approximation, u_expert is expected feature counts
        c_q = np.concatenate((np.dot(-posterior_rewards.transpose(), u_expert), p0))  #for the auxiliary variables z
    else:
        c_q = np.concatenate((np.dot(-R.transpose(), u_expert), p0))  #for the auxiliary variables z

    #constraints: 

    #sum of q's should equal 1
    A_eq = np.concatenate((np.ones((1,n)), np.zeros((1,num_states))), axis = 1)
    b_eq = np.ones(1)
    
    #leq constraints

    #first do the q <= 1/(1-alpha) p
    A_q_leq_p = np.concatenate((np.eye(n), np.zeros((n, num_states))), axis=1)
    b_q_leq_p = 1.0/(1 - alpha) * p_R

    #next do the value iteration equations
    I_s = np.eye(num_states)
    #TODO: debug this and check it is more general using Ps (see cvar method)
    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a)

    trans_dyn = np.concatenate(I_minus_gamma_Ps, axis=0)

    # if mdp_env.num_actions == 4:
    #     trans_dyn = np.concatenate(( I_s - gamma * mdp_env.P_left,
    #                             I_s - gamma * mdp_env.P_right,
    #                             I_s - gamma * mdp_env.P_up,
    #                             I_s - gamma * mdp_env.P_down), axis=0)
    # else:
    #     trans_dyn = np.concatenate((I_s - gamma * mdp_env.P_left,
    #                            I_s - gamma * mdp_env.P_right), axis=0)
    
    A_vi = np.concatenate((R, -trans_dyn), axis=1)
    b_vi = np.zeros(num_states * num_actions)

    #last add constraint that all q >= 0
    A_q_geq_0 = np.concatenate((-np.eye(n), np.zeros((n, num_states))), axis=1)
    b_q_geq_0 = np.zeros(n)

    #stick them all together
    A_leq = np.concatenate((A_q_leq_p,
                            A_vi,
                            A_q_geq_0), axis=0)
    b_geq = np.concatenate((b_q_leq_p, b_vi, b_q_geq_0))

    #solve the LP
    sol = linprog(c_q, A_eq=A_eq, b_eq=b_eq, A_ub=A_leq, b_ub=b_geq, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    #print("solution to optimizing for CVaR reward")
    #print(sol)
    cvar = sol['fun'] #I think the objective value should be the same?
    #the solution of the LP corresponds to the CVaR
    q = sol['x'][:n] #get sigma (this is VaR (at least close))
    values = sol['x'][n:]
    print("CVaR = ", cvar)
    print("policy v(s) under Rq = ", values)
    print("expected value", np.dot(mdp_env.init_dist, values))
    
    #print("q weights:", q)
    cvar_reward_fn = np.dot(R,q)
    #print("CVaR reward Rq =", cvar_reward_fn)

    return cvar_reward_fn, q


def get_policy_rewards(stoch_pi, rewards_sa):
    num_states, num_actions = stoch_pi.shape
    policy_rewards = np.zeros(num_states)
    for s, a_probs in enumerate(stoch_pi):
        expected_reward = 0.0
        for a, prob in enumerate(a_probs):
            index = s + num_states * a
            expected_reward += prob * rewards_sa[index]
        policy_rewards[s] = expected_reward
    return policy_rewards


#TODO: might be able to vectorize this and speed it up!
def get_policy_transitions(stoch_pi, mdp_env):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    P_pi = np.zeros((num_states, num_states))
    #calculate expectations
    for s1 in range(num_states):
        for s2 in range(num_states):
            cum_prob = 0.0
            for a in range(num_actions):
                cum_prob += stoch_pi[s1,a] * mdp_env.get_transition_prob(s1,a,s2)
            P_pi[s1,s2] =  cum_prob
    return P_pi

def get_policy_state_occupancy_frequencies(stoch_policy, mdp_env):
    P_pi = get_policy_transitions(stoch_policy, mdp_env)
    A = np.eye(mdp_env.get_num_states()) - mdp_env.gamma * P_pi.transpose()
    return np.linalg.solve(A, mdp_env.init_dist)
    
def get_policy_expected_return(stoch_policy, mdp_env):
    u_pi = get_policy_state_occupancy_frequencies(stoch_policy, mdp_env)
    R_pi = get_policy_rewards(stoch_policy, mdp_env.r_sa)
    return np.dot(u_pi, R_pi)

def stoch_policy_to_usa(stoch_policy, mdp_env):
    '''Transform a stochastic policy into state-action expected occupancies
    '''
    n_states = mdp_env.num_states
    u_s = get_policy_state_occupancy_frequencies(stoch_policy, mdp_env)
    u_sa = np.zeros(mdp_env.num_states * mdp_env.num_actions)
    for s in range(mdp_env.num_states):
        for a in range(mdp_env.num_actions):
            u_sa[s + a * n_states] = u_s[s] * stoch_policy[s,a]
    return u_sa


def two_by_two_mdp():
    num_rows = 2
    num_cols = 2
    r_s = np.array( [-1, -10, -1, 1] )
    gamma = 0.5
    init_dist = np.array([0.5,0.5,0,0])
    mdp = BasicGridMDP(num_rows, num_cols, r_s, gamma, init_dist)
    u_sa_opt = solve_mdp_lp(mdp)
    print(u_sa_opt)
    print("u[state,actions]=")
    print(np.reshape(u_sa_opt, (num_rows*num_cols, 4)).transpose())
    pi_opt = utils.get_optimal_policy_from_usa(u_sa_opt, mdp)
    print("optimal policy")
    print("pi(a|s) for left, right, up, down")
    print(pi_opt)


def two_state_chain():
        
    num_states = 2
    num_actions = 2
    gamma = 0.5
    p0 = np.array([0.5,0.5])
    r_sa = np.array([1,-1, +1, -1])
    chain = ChainMDP(num_states, r_sa, gamma, p0)
    #u_sa_opt = solve_mdp_lp(chain)  #np.zeros(np.shape(r_sa))
    u_sa_opt = np.zeros(np.shape(r_sa))
    print(u_sa_opt)
    print("u[state,actions]=")
    print(np.reshape(u_sa_opt, (num_states, num_actions)).transpose())
    pi_opt = utils.get_optimal_policy_from_usa(u_sa_opt, chain)
    print("optimal policy")
    print("pi(a|s) for left, right")
    print(pi_opt)
    print('optimizing CVAR policy')
    R1 = np.array([0., +1., 0., 0.])
    R2 = np.array([0.5, 0, 0.5, 0])
    R3 = np.array([0.5, 0.5, 0., 0.])
    R = np.vstack((R1,R2,R3)).transpose()  #each reward hypothesis is a column so stack as rows then transpose
    print("reward posterior")
    print(R)
    k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    posterior_probs = np.array([1/3, 1/3, 1/3])
    #we need to add auxiliary variables to get the ReLU in the objective.

    alpha = 0.5 #TODO:change back  #(1-alpha)% average worst-case. 

    cvar_opt_usa = solve_max_cvar_policy(chain, u_sa_opt, R, posterior_probs, alpha)

    u_expert = u_sa_opt
    policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)




    #TODO: check below to see if it is correct.
    print("let's check the solution")
    #first we need the u^Tri terms for each reward in the posterior. Turns out I already computed them above.

    c_ploss = policy_losses  #multiply by negative since we min in the lp solver

    _,num_rew_samples = R.shape

    #add constraint q <= 1/(1-alpha) * p 
    A = np.eye(num_rew_samples)
    b = 1 / (1-alpha) * posterior_probs 
    #and add constraint that sum q = 1
    A_eq = np.ones((1,num_rew_samples))
    b_eq = np.ones(1)

    #print(A)
    #print(b)
    #solve the LP
    sol = linprog(c_ploss, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq) #use default bounds since they are set to (0,infty)
    print("solving robust form to get cvar worst-case reward")
    print(sol)
    q_star = sol['x']
    print("Cvar robust", sol['fun'])
    R_cvar = np.dot(R, q_star)
    print("CVaR reward is", R_cvar)
    cvar2 = np.dot(cvar_opt_usa, R_cvar) - np.dot(u_expert, R_cvar)
    print("alternative CVaR calculation yields", cvar2)
    cvar3 = np.dot(q_star, np.dot(R.transpose(), cvar_opt_usa - u_expert))
    print("another CVaR calc", cvar3)

    print("solving for optimal policy for CVaR reward")
    new_mdp = ChainMDP(num_states, R_cvar, gamma, p0)
    new_pi = solve_mdp_lp(new_mdp)
    print("new opt policy")
    print(new_pi)
    print(utils.get_optimal_policy_from_usa(new_pi, new_mdp))

    print(utils.print_table_row(np.concatenate((R_cvar, q_star))))

    cvar_r_dual, q_dual = solve_minCVaR_reward(chain, u_expert, R, posterior_probs, alpha)

    #okay so let's check and see what happens if we optimize cvar_r_dual
    dual_reward_mdp = ChainMDP(num_states, cvar_r_dual, gamma, p0)
    dual_opt_pi = solve_mdp_lp(new_mdp)
    print("dual opt policy")
    print(dual_opt_pi)
    print(utils.get_optimal_policy_from_usa(dual_opt_pi, dual_reward_mdp))
    #interesting, this doesn't give the same policy either...I guess this makes sense since we're taking a game and then 
    #we're taking away one player so the other player (in this case the policy optimizer) should get higher value since it can play
    # a best response. Kind of like how the GAIL reward function isn't good by itself but only works when optimized with the agent.

    #The CVaR's match so that's good.

    #does this reward match the reward found by taking 


def value_iteration(mdp_env, epsilon=0.0001):
    '''standard value iteration'''
    gamma = mdp_env.gamma
    num_states = mdp_env.num_states
    num_actions = mdp_env.num_actions
    Ps = mdp_env.Ps
    #repeat until convergence within error eps
    V = np.zeros(mdp_env.num_states)
    while True:
        delta = 0
        for s1 in range(num_states):
        
            tempV = 0
            #add reward
            tempV += mdp_env.r_s[s1]
        
            #add discounted max over actions of value of next state
            maxActionValue = -np.inf
            for a in range(num_actions):
                T = Ps[a]
                #calculate expected utility of taking action a in state s1
                expUtil = 0
                for s2 in range(num_states):
                    expUtil += T[s1][s2] * V[s2]
                
                if expUtil > maxActionValue:
                    maxActionValue = expUtil
            
            tempV += gamma * maxActionValue

            #update delta to track convergence
            absDiff = abs(tempV - V[s1])
            
            if absDiff > delta:
                delta = absDiff
            V[s1] = tempV
        if delta < epsilon * (1 - gamma) / gamma:
            return V

def calculate_Q_values(mdp_env, V = None):
    if V is None:
        V = value_iteration(mdp_env)
    Q = np.zeros((mdp_env.num_states, mdp_env.num_actions))
    for s in range(mdp_env.num_states):
        for a in range(mdp_env.num_actions):
            Q[s][a] = mdp_env.r_s[s]
            for s2 in range(mdp_env.num_states):
                Q[s][a] += mdp_env.gamma * mdp_env.Ps[a][s][s2] * V[s2]
    return Q    

          
def get_optimal_stochastic_policy(mdp_env, Q = None, V = None, epsilon = 0.0001):
    num_states, num_actions = mdp_env.num_states, mdp_env.num_actions
    opt_stochastic = np.zeros((num_states, num_actions))
    if Q is None:
       Q = calculate_Q_values(mdp_env, V)

    for s in range(num_states):
        cum_sum = 0.0
        max_qval = np.max(Q[s])
        for a in range(num_actions):

            if(abs(Q[s,a] - max_qval) < epsilon):            
                opt_stochastic[s][a] = 1.0
                cum_sum += 1.0     
        
        #normalize
        opt_stochastic[s] /= cum_sum
   
    return opt_stochastic

def soft_value_iteration_old(mdp_env, epsilon=0.0001):
    '''standard value iteration
    Note: should work with state and state-action rewards
    This method works without a terminal state, but not with one...
    '''
    gamma = mdp_env.gamma
    num_states = mdp_env.num_states
    num_actions = mdp_env.num_actions
    Ps = mdp_env.Ps
    #repeat until convergence within error eps
    V1 = np.zeros(mdp_env.num_states)
    
    while True:
        Q = np.zeros((mdp_env.num_states, mdp_env.num_actions))
        V = V1.copy()
        delta = 0
        for s1 in range(num_states):
            for a in range(num_actions):
                #calculate expected utility of taking action a in state s1
                exp_util = mdp_env.r_s[s1]# + a*num_states] 
                T = Ps[a]
                for s2 in range(num_states):
                    print(gamma * T[s1][s2] * V[s2])
                    exp_util += gamma * T[s1][s2] * V[s2]
                Q[s1][a] = exp_util
            #rather than using max use softmax
            V1[s1] = utils.logsumexp(Q[s1])

            #update delta to track convergence
            delta = max(delta, (V1[s1] - V[s1]))
            
        if delta < epsilon * (1 - gamma) / gamma:
            return V, Q


def ziebart_softmax(x1, x2):
    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))


#TODO: debug this. It isn't working....
def soft_value_iteration(mdp_env, epsilon=0.0001):
    '''Using notes from Brian Ziebart's thesis on page 112
    Note: should work with state and state-action rewards
    Not working well, maybe I need to try without a terminal? that works.
    Now what?
    '''
    gamma = mdp_env.gamma
    num_states = mdp_env.num_states
    num_actions = mdp_env.num_actions
    Ps = mdp_env.Ps
    #repeat until convergence within error eps
    Vsoft = np.zeros(mdp_env.num_states)
    for s in range(num_states):
        Vsoft[s] = -1000000
    
    
    while True: #while not converged

        #set with potential function see brief discussion on page 111
        Vsoft_prime = np.zeros(mdp_env.num_states)
        for s in range(mdp_env.num_states):
            if s not in mdp_env.terminals:
                Vsoft_prime[s] = -1000000

        Qsoft = np.zeros((mdp_env.num_states, mdp_env.num_actions))
        
        delta = 0
        for s1 in range(num_states):
            for a in range(num_actions):
                #calculate expected utility of taking action a in state s1
                T = Ps[a]
                exp_util = mdp_env.r_s[s1]
                for s2 in range(num_states):
                    exp_util += gamma * T[s1][s2] * Vsoft[s2]

                Qsoft[s1][a] = exp_util
                Vsoft_prime[s1] = ziebart_softmax(Vsoft_prime[s1], Qsoft[s1][a])
            
            #update delta to track convergence
            print("state ", s1)
            print(abs(Vsoft[s1] - Vsoft_prime[s1]))
            delta = max(delta, abs(Vsoft[s1] - Vsoft_prime[s1]))
        Vsoft = Vsoft_prime.copy()   
        print(Vsoft) 
        if delta < epsilon * (1 - gamma) / gamma:
            return Vsoft, Qsoft



def compute_maxent_policy(mdp_env, beta, epsilon=0.0001):
    _, Q = soft_value_iteration(mdp_env, epsilon)
    stoch_policy = np.zeros((mdp_env.num_states, mdp_env.num_actions))
    for s in range(mdp_env.num_states):
        stoch_policy[s,:] = utils.stable_softmax(Q[s]) 
    return stoch_policy

if __name__ == "__main__":
    # two_state_chain()
    import mdp_worlds
    mdp_env = mdp_worlds.lava_ambiguous_corridor()
    #mdp_env = mdp_worlds.machine_teaching_toy_featurized()
    beta = 1
    #V = value_iteration(mdp_env)
    stoch_policy = get_optimal_stochastic_policy(mdp_env)
    utils.print_stoch_policy(stoch_policy, mdp_env)
    soft_V, soft_Q = soft_value_iteration(mdp_env)

    print("soft V")
    print(soft_V)
    print("soft Q")
    print(soft_Q)
    stoch_pi = compute_maxent_policy(mdp_env, beta) 
    utils.print_stoch_policy(stoch_pi, mdp_env)
    print(np.sum(stoch_pi, axis=1))


