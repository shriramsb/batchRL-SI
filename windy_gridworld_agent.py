import numpy as np
from agent import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class Network(nn.Module):
	"""
	Models Q-value as a neural network with inputs s, a and a scalar output Q(s, a)
	"""
	def __init__(self, input_dim, dropout_input=0.0, dropout_hidden=0.0):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(input_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 1)
		self.state = {'dropout_input' : dropout_input, 'dropout_hidden' : dropout_hidden}
		self.state.update({'is_training' : True})

	def updateState(self, val):
		self.state.update(val)

	def forward(self, x):
		dropout_input = self.state['dropout_input']
		dropout_hidden = self.state['dropout_hidden']
		is_training = self.state['is_training']
		x = F.dropout(x, p=dropout_input, training=is_training)
		x = F.dropout(F.relu(self.fc1(x)), p=dropout_hidden, training=is_training)
		x = F.dropout(F.relu(self.fc2(x)), p=dropout_hidden, training=is_training)
		x = F.dropout(self.fc3(x), p=dropout_hidden, training=is_training)
		return x

class BatchRLAgent(Agent):
	def __init__(self, ER_epochs, episodes_per_batch, epsilon, gamma, learning_hparams):
		self.ER_epochs = ER_epochs 						# number of epochs to train in batchER
		self.episodes_per_batch = episodes_per_batch 	# number of episodes to run before training on batch
		self.epsilon = epsilon 	 						# epsilon for epsilon-greedy action selection
		self.gamma = gamma 								# decay of reward in environment

		self.learning_hparams = learning_hparams 		# has lr, momentum, batch_size

		# Q to be approximated by a neural network
		self.Q = None             # initialize pytorch neural network

		self.D = []                 # all experiences
		self.D.append([]) 			
		self.episode = 0 			# keeps track of episode number 

	def initQNetwork(self, state, actions, dropout_input, dropout_hidden):
		"""
			Initialize Q network with appropriate number of inputs, according to encoding used.
			Currently uses one-hot encoding for each dimension of state, action. Encoded vectors are concatenated to get input to NN
			MSE loss with target Q value and SGD+momentum optimizer used
		"""
		self.state_encoding_sizes = np.array(state.max_pos) - np.array(state.origin) + 1
		self.state_encoding_dim = np.sum(self.state_encoding_sizes)
		self.state_encoding_sizes = tuple(self.state_encoding_sizes)
		self.action_encoding_dim = len(actions)
		self.actions_to_index = {}
		self.actions_to_index = {actions[i] : i for i in range(len(actions))}
		self.encoding_dim = self.state_encoding_dim + self.action_encoding_dim
		
		self.Q = Network(self.encoding_dim, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
		self.mse_loss = nn.MSELoss()
		self.optim = torch.optim.SGD(self.Q.parameters(), lr=self.learning_hparams['learning_rate'], momentum=self.learning_hparams['momentum'])

	def getAction(self, state, train=False):
		"""
			If train=True, returns epsilon-greedy action w.r.t Q.
			If train=False, return optimal action w.r.t Q.
		"""

		self.Q.updateState({'is_training' : False})

		actions_list = state.getLegalActions()
		if (train):
			# return random action with probability self.epsilon
			if (np.random.random() < self.epsilon):
				return actions_list[np.random.choice(range(len(actions_list)))]

		# find optimal action
		sa_list = list(itertools.product([state], actions_list))
		encoding = self.getEncodingsFromList(sa_list)
		with torch.no_grad():
			output = self.Q(torch.Tensor(encoding))
			optimal_action_index = torch.argmax(torch.squeeze(output, dim=1), dim=0).item()
		return actions_list[optimal_action_index]

	def update(self, d, is_episode_end):
		"""
			Update agent's experience. After sufficient experiences, runs ER or FQI to update Q values
		"""
		self.D[self.episode].append(d)
		if (is_episode_end):
			self.D.append([])
			self.episode += 1
		
		if (is_episode_end and (self.episode % self.episodes_per_batch == 0)):
			# train the network with experience of current batch
			# can either use ER or FQI
			self.trainQNetworkER()

	def trainQNetworkER(self):
		"""
			Trains Q-Network with experience replay with only the last batch of data. 
			Replays examples in reverse order
		"""
		batch_size = self.learning_hparams['batch_size']
		data = self.D[-2][: : -1] 			# Reversing last batch of data. (-2) since an empty list is appended to self.D at the end of episode
		# Train for self.ER_epochs
		for epoch in range(self.ER_epochs):
			pos = 0
			epoch_done = False

			# Updates corresponding to single epoch
			while (True):
				
				# extract batch of data
				if (pos + batch_size >= len(data)):
					epoch_done = True
				
				batch = data[pos : pos + batch_size]
				pos += batch_size

				# get list of next states in batch
				batch_next_states = [v[3] for v in batch]

				# get max_{a}Q(next state, a)
				with torch.no_grad():
					max_Q = []
					for i in range(len(batch_next_states)):
						best_action = self.getAction(batch_next_states[i], train=False)
						max_q = self.Q(torch.Tensor(self.getEncodingsFromList([(batch_next_states[i], best_action)])))
						max_Q.append(max_q.item())

				# get target r + max_Q
				target = np.array([v[2] for v in batch]) + self.gamma * np.array(max_Q)

				# list of (s, a) tuples in batch
				batch_inputs = [(v[0], v[1]) for v in batch]
				batch_inputs_encoded = self.getEncodingsFromList(batch_inputs)

				# train step
				outputs = self.Q(torch.Tensor(batch_inputs_encoded))
				loss = self.mse_loss(outputs, torch.unsqueeze(torch.Tensor(target), dim=1))
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()

				if (epoch_done):
					break
		

	def getEncodingsFromList(self, sa_list):
		encodings = np.empty((len(sa_list), self.encoding_dim), dtype=np.float32)
		for i in range(len(sa_list)):
			sa = sa_list[i]
			encodings[i] = np.concatenate((self.getStateEncoding(sa[0]), self.getActionEncoding(sa[1])), axis=0)
		return encodings

	def getStateEncoding(self, state):
		"""
			Returns one-hot encoding of state as 1D numpy array, by appending one-hot vectors for x and y coordinates. 
		"""
		encoding = []
		origin = state.origin
		pos = state.pos
		for i in range(len(pos)):
			cur_encoding = [0.0 for _ in range(self.state_encoding_sizes[i])]
			cur_encoding[pos[i] - origin[i]] = 1.0
			encoding.extend(cur_encoding)
		return np.array(encoding)

	def getActionEncoding(self, action):
		"""
			Returns one-hot encoding of action as 1D numpy array.
		"""
		encoding = [0.0 for _ in range(self.action_encoding_dim)]
		encoding[self.actions_to_index[action]] = 1.0
		return np.array(encoding)        

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

