import numpy as np
from agent import *


class BatchRLAgent(Agent):
    def __init__(self, ER_iters, episodes_per_batch, batch_size, epsilon):
        self.ER_iters = ER_iters
        self.episodes_per_batch = episodes_per_batch
        self.batch_size = batch_size
        self.epsilon = epsilon

        # Q to be approximated by a neural network
        self.net = None             # initialize pytorch neural network

        self.D = []                 # all experiences
        self.D.append([])
        self.episode = 0

    def getAction(self, state, train=False):
        if (train):
            pass
            # evaluate self.net at state and all possible actions at current state. Take the one with max. Q with epsilon probability
        else:
            pass
            # evaluate self.net at state and all possible actions at current state. Take the one with max. Q

    def update(self, d, is_episode_end):
        
        self.D[self.episode].append(d)
        if (is_episode_end):
            self.D.append([])
            self.episode += 1
        
        if (is_episode_end and (self.episode % self.episodes_per_batch == 0)):
            # train the network with experience of current batch
            # can either use ER or FQI
            pass
        

# Implementation of Sarsa agent

# class SarsaAgent(object):
#     def __init__(self, problem, alpha=0.2, epsilon=0.1):
#         self.problem = problem                  # MDP
#         states_list = problem.getAllStates()
#         actions_list = problem.getAllActions()
        
#         self.Q = {}                             # Q values
#         Q = self.Q
#         # populating Q values with 0
#         for state in states_list:
#             Q[state] = []
#             for action in actions_list:
#                 Q[state].append((action, 0.0))
#             Q[state] = np.array(Q[state])

#         # index of given action in Q[state] list
#         self.action_to_index = {}
#         for i, action in enumerate(actions_list):
#             self.action_to_index[action] = i
        
#         # parameters for learning
#         self.alpha = alpha
#         self.epsilon = epsilon
    
#     # trains for a given number of episodes ; number of steps spent in each episode
#     def train(self, num_episodes):
#         Q = self.Q
#         problem = self.problem
#         steps_per_episode = []
#         for _ in range(num_episodes):
#             step = 0
#             problem.resetCurrentState()                     # set current state to start state
#             state = problem.getStartState()                 # get current state
#             action = self.getEpsilonGreedyAction(state)     # choose action epsilon-greedily

#             while (True):
#                 reward, next_state, is_terminal = problem.getNextStateReward(action)    # getting reward, next state from MDP
#                 step += 1
#                 next_action = self.getEpsilonGreedyAction(next_state)                   # choose next action epsilon-greedily

#                 # update Q(s, a)
#                 new_Q = self.getQ(state, action) + self.alpha * (reward + problem.gamma * self.getQ(next_state, next_action) - self.getQ(state, action))
#                 self.setQ(state, action, new_Q)
                
#                 # update current state and action to perform
#                 state, action = next_state, next_action
                
#                 if (is_terminal):
#                     break
            
#             steps_per_episode.append(step)
#         return steps_per_episode

#     def getQ(self, state, action):
#         return self.Q[state][self.action_to_index[action]][1]

#     def setQ(self, state, action, val):
#         self.Q[state][self.action_to_index[action]][1] = val

#     def resetQ(self, state, action):
#         Q = self.Q
#         for k, v in Q.items():
#             Q[k][:, 1] = 0.0

#     # returns epsilon-greedy action using current Q values for given state
#     def getEpsilonGreedyAction(self, state):
#         if (np.random.rand() < self.epsilon):
#             action = self.getRandomAction(state)
#         else:
#             action = self.getOptimalAction(state)
#         return action
        
#     def getRandomAction(self, state):
#         return np.random.choice(self.Q[state][:, 0])
    
#     def getOptimalAction(self, state):
#         return self.Q[state][np.argmax(self.Q[state][:, 1]), 0]


#     def getEncoding(self):
#         """
#             Returns one-hot encoding as 1D numpy array, by appending one-hot vectors for x and y coordinates. 
#             Make sure type(self)._encoding_sizes is filled before calling this.
#         """
#         encoding = []
#         origin = type(self)._origin
#         pos = self.pos
#         for i in range(len(pos)):
#             cur_encoding = [0.0 for _ in range(type(self)._encoding_sizes[i])]
#             cur_encoding[pos[i] - origin[i]] = 1.0
#             encoding.extend(cur_encoding)
#         return np.array(encoding)

#     def getEncoding(self):
#         """
#             Returns one-hot encoding as 1D numpy array, by appending one-hot vectors for x and y coordinates. 
#             Make sure type(self)._encoding_sizes is filled before calling this.
#         """
#         encoding = []
#         direction = self.direction
#         for i in range(len(direction)):
#             cur_encoding = [0.0 for _ in range(type(self)._encoding_sizes[i])]
#             cur_encoding[pos[i]] = 1.0
#             encoding.extend(cur_encoding)
#         return np.array(encoding)

