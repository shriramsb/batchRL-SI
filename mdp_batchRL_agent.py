import numpy as np
from agent import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import sys
import util
from copy import deepcopy
import math

import mdp_environment

import utils

class Network(nn.Module):
	"""
	Fully-connected feedforward neural network
	"""
	def __init__(self, input_dim, output_dim, dropout_input=0.0, dropout_hidden=0.0, num_layers=1, hidden_dim=20):
		super(Network, self).__init__()
		self.num_layers = num_layers
		self.fc = nn.ModuleList()
		self.layer_input_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
		for i in range(num_layers):
			self.fc.append(nn.Linear(self.layer_input_dims[i], self.layer_input_dims[i + 1]))
		self.state = {'dropout_input' : dropout_input, 'dropout_hidden' : dropout_hidden}
		self.state.update({'is_training' : True})

	def updateState(self, val):
		self.state.update(val)

	def forward(self, x):
		dropout_input = self.state['dropout_input']
		dropout_hidden = self.state['dropout_hidden']
		is_training = self.state['is_training']
		x = F.dropout(x, p=dropout_input, training=is_training)
		for i in range(self.num_layers - 1):
			x = F.dropout(F.relu(self.fc[i](x)), p=dropout_hidden, training=is_training)
		x = self.fc[self.num_layers - 1](x)
		return x

class BatchRLAgent(Agent):
	def __init__(self, ER_epochs, episodes_per_batch, epsilon, gamma, learning_hparams, 
					multi_output=False, 
					gpu_id=-1, 
					encoding_type='one-hot'):
		self.ER_epochs = ER_epochs 						# number of epochs to train in batchER
		self.episodes_per_batch = episodes_per_batch 	# number of episodes to run before training on batch
		self.epsilon = epsilon 	 						# epsilon for epsilon-greedy action selection
		self.gamma = gamma 								# decay of reward in environment

		self.learning_hparams = learning_hparams 		# has lr, momentum, batch_size

		self.multi_output = multi_output 				# if multi_output=True, then network takes only s as input and gives Q(s, a) for all actions a
		self.gpu_id = gpu_id 							# If gpu_id!=-1, then uses gpu to train network

		# Q to be approximated by a neural network
		self.Q = None

		self.D = []                 # all experiences ; D[i] contains list of experiences in ith batch of batchRL
		self.D.append([]) 			

		self.episode = 0 			# keeps track of episode number
		self.encoding_type = encoding_type
		mdp_environment.MDPState.initEncoder(encoding_type)

		self.importance = None 


	def initQNetwork(self, state, actions, dropout_input, dropout_hidden, num_layers=1, hidden_dim=20):
		"""
			Initialize Q network with appropriate number of inputs, according to encoding used.
			Currently uses one-hot encoding for each dimension of state, action. Encoded vectors are concatenated to get input to NN
			MSE loss with target Q value and SGD+momentum optimizer used
		"""
		self.state_encoding_dim = state.encoding_dim
		if (self.encoding_type == 'one-hot'):
			self.action_encoding_dim = len(actions)
		elif (self.encoding_type == 'binary'):
			self.action_encoding_dim = math.ceil(math.log(len(actions), 2))
		self.actions_to_index = {actions[i] : i for i in range(len(actions))}
		self.index_to_action = {i : actions[i] for i in range(len(actions))}
		self.encoding_dim = self.state_encoding_dim + self.action_encoding_dim
		
		if (not self.multi_output):
			self.Q = Network(self.encoding_dim, 1, dropout_input=dropout_input, dropout_hidden=dropout_hidden, 
								num_layers=num_layers, hidden_dim=hidden_dim)
		else:
			self.Q = Network(self.state_encoding_dim, len(actions), dropout_input=dropout_input, dropout_hidden=dropout_hidden, 
								num_layers=num_layers, hidden_dim=hidden_dim)
		self.mse_loss = nn.MSELoss()
		self.optim = torch.optim.SGD(self.Q.parameters(), lr=self.learning_hparams['learning_rate'], momentum=self.learning_hparams['momentum'])

		self.fast_device = torch.device('cpu')
		if (self.gpu_id >= 0):
			if (not torch.cuda.is_available()):
				print("GPU not detected ; using CPU")
			else:
				self.fast_device = torch.device('cuda:' + str(self.gpu_id))

		self.cpu_device = torch.device('cpu')
		self.Q.to(self.fast_device)

		params = list(self.Q.parameters())
		#print("Initial Params: {}".format(params))
		self.importance = utils.get_zero_like(params)

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
		#doesn't use GPU if not using multiple_outputs
		if (not self.multi_output):
			sa_list = list(itertools.product([state], actions_list))
			encoding = self.getEncodingsFromList(sa_list)
			with torch.no_grad():
				output = self.Q(torch.from_numpy(encoding))
				optimal_action_index = torch.argmax(torch.squeeze(output, dim=1), dim=0).item()
		else:
			state_encoding = self.getStateListEncoding([state])
			state_encoding = torch.from_numpy(state_encoding).to(self.fast_device)
			with torch.no_grad():
				output = self.Q(state_encoding)
				output = torch.squeeze(output, dim=0)
			legal_actions_one_hot = np.zeros(self.action_encoding_dim, dtype=np.float32)
			for a in actions_list:
				legal_actions_one_hot[self.actions_to_index[a]] = 1
			legal_actions_one_hot = torch.from_numpy(legal_actions_one_hot).to(self.fast_device)
			output = output + (-1e24) * (1 - legal_actions_one_hot)
			optimal_action_index = torch.argmax(output, dim=0).item()

		return actions_list[optimal_action_index]



	def update(self, d, is_episode_end):
		"""
			Update agent's experience. After sufficient experiences, runs ER or FQI to update Q values
		"""
		self.D[-1].append(d)
		if (is_episode_end):
			self.episode += 1
		
		if (is_episode_end and (self.episode % self.episodes_per_batch == 0)):
			self.D.append([])
			# train the network with experience of current batch
			# can either use ER or FQI
			self.trainQNetworkER()

	def trainQNetworkER(self):
		"""
			Trains Q-Network with experience replay with only the last batch of data. 
			Replays examples in reverse order
		"""
		initial_params_value = utils.get_params_data(list(self.Q.parameters()))
		pi_accumulator = utils.get_zero_like(list(self.Q.parameters()))
		batch_size = self.learning_hparams['batch_size']
		data = self.D[-2][: : -1] 			# Reversing last batch of data. (-2) since an empty list is appended to self.D at the end of batch just before calling this function
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

				# GPU not used if not using multi_output
				if (not self.multi_output):
					# get max_{a}Q(next state, a)
					with torch.no_grad():
						max_Q = []
						for i in range(len(batch_next_states)):
							if (batch_next_states[i].is_terminal):
								max_Q.append(0.0)
								continue
							best_action = self.getAction(batch_next_states[i], train=False)
							max_q = self.Q(torch.from_numpy(self.getEncodingsFromList([(batch_next_states[i], best_action)])))
							max_Q.append(max_q.item())

					# get target r + max_Q
					target = np.array([v[2] for v in batch], dtype=np.float32) + self.gamma * np.array(max_Q, dtype=np.float32)

					# list of (s, a) tuples in batch
					batch_inputs = [(v[0], v[1]) for v in batch]
					batch_inputs_encoded = self.getEncodingsFromList(batch_inputs)

					# train step
					self.Q.updateState({'is_training' : True})
					outputs = self.Q(torch.from_numpy(batch_inputs_encoded))
					loss = self.mse_loss(outputs, torch.unsqueeze(torch.from_numpy(target), dim=1))
					self.optim.zero_grad()
					loss.backward(retain_graph=True)
					#print("loss: {}".format(loss))

					params = list(self.Q.parameters())
					initial_parameters = utils.get_params_data(params)
					gradients = utils.get_grads_from_params(params)

					regularised_loss = loss + self.learning_hparams['reg'] * utils.get_regularisation_penalty(params, initial_params_value, self.importance)
					self.optim.zero_grad()
					regularised_loss.backward()
					self.optim.step()
					

					final_parameters = utils.get_params_data(list(self.Q.parameters()))

					delta_parameters = utils.sub_tensor_lists(final_parameters, initial_parameters)
					pi_component = utils.delta_param_gradient_product(delta_parameters, gradients)
					utils.add_to_tensor_lists(pi_accumulator, pi_component)

					self.Q.updateState({'is_training' : False})

				else:

					# find max_{a}Q(next state, a)
					batch_next_states_fd = torch.from_numpy(self.getStateListEncoding(batch_next_states)).to(self.fast_device)
					with torch.no_grad():
						next_state_Q = self.Q(batch_next_states_fd)
						for i in range(len(batch_next_states)):
							if (batch_next_states[i].is_terminal):
								next_state_Q[i] *= 0.0
					actions_one_hot = np.zeros((len(batch_next_states), self.action_encoding_dim), dtype=np.float32)
					for i in range(len(batch_next_states)):
						actions = batch_next_states[i].getLegalActions()
						for a in actions:
							actions_one_hot[i, self.actions_to_index[a]] = 1

					actions_one_hot = torch.from_numpy(actions_one_hot).to(self.fast_device)
					next_state_Q = self.gamma * next_state_Q + (-1e24) * (1 - actions_one_hot)
					next_state_Q_max = torch.max(next_state_Q, dim=1)[0]

					# get targets ; r + max_Q
					target = torch.from_numpy(np.array([v[2] for v in batch], dtype=np.float32)).to(self.fast_device) + next_state_Q_max


					# list of s tuples in batch
					self.Q.updateState({'is_training' : True})
					batch_inputs = [v[0] for v in batch]
					batch_inputs_encoded = torch.from_numpy(self.getStateListEncoding(batch_inputs)).to(self.fast_device)
					outputs_all = self.Q(batch_inputs_encoded)
					actions_one_hot = np.zeros((len(batch_inputs), self.action_encoding_dim), dtype=np.float32)
					for i in range(len(batch_next_states)):
						actions_one_hot[i, self.actions_to_index[batch[i][1]]] = 1
					actions_one_hot = torch.from_numpy(actions_one_hot).to(self.fast_device)
					outputs = torch.max(outputs_all + (-1e24) * (1 - actions_one_hot), dim=1)[0]
					loss = self.mse_loss(outputs, target)
					self.optim.zero_grad()
					loss.backward(retain_graph=True)

					params = list(self.Q.parameters())
					initial_parameters = utils.get_params_data(params)
					
					gradients = utils.get_grads_from_params(params)

					#sys.exit(0)

					regularised_loss = loss + self.learning_hparams['reg'] * utils.get_regularisation_penalty(params, initial_params_value, self.importance)
					self.optim.zero_grad()
					regularised_loss.backward()
					self.optim.step()

					final_parameters = utils.get_params_data(list(self.Q.parameters()))
					#print("final_parameters: {}".format(final_parameters))
					#print("initial_parameters: {}".format(initial_parameters))

					delta_parameters = utils.sub_tensor_lists(final_parameters, initial_parameters)
					pi_component = utils.delta_param_gradient_product(delta_parameters, gradients)
					#print("pi_component: {}".format(pi_component))
					utils.add_to_tensor_lists(pi_accumulator, pi_component)
					self.Q.updateState({'is_training' : False})

				if (epoch_done):
					break
		
		final_params_value = utils.get_params_data(list(self.Q.parameters()))
		delta_params = utils.sub_tensor_lists(final_params_value,initial_params_value)
		utils.update_importance(self.importance, pi_accumulator, delta_params, self.training_hparams['importance_retain_factor'])


	def getEncodingsFromList(self, sa_list):
		encodings = np.empty((len(sa_list), self.encoding_dim), dtype=np.float32)
		for i in range(len(sa_list)):
			sa = sa_list[i]
			encodings[i] = np.concatenate((self.getStateEncoding(sa[0]), self.getActionEncoding(sa[1])), axis=0)
		return encodings

	def getQValue(self, state, action):
		"""
			Given state, action, get Q-value from current network state
		"""
		if (not self.multi_output):
			return self.Q(torch.Tensor(self.getEncodingsFromList([(state, action)])).to(self.fast_device)).item()
		else:
			return torch.squeeze(self.Q(torch.Tensor(self.getStateListEncoding([state])).to(self.fast_device)), dim=0)[self.actions_to_index[action]].item()

	def getQValuesFromStateList(self, state_list):
		"""
			Gets Q value for all legals actions for each state in the state list. 
			Currently inefficient due to doing 1 forward pass for each (state, action) pair.
			TODO : make required pairs into batch and find Q-values in few forward passes
		"""
		q_val = []
		for state in state_list:
			q_val.append({})
			for a in state.getLegalActions():
				q_val[-1][a] = self.getQValue(state, a)

		return q_val

	def getParameterValues(self):
		p_numpy = []
		for p in self.Q.parameters():
			p_numpy.append(deepcopy(p.to(self.cpu_device).detach().numpy()))
		return p_numpy

	def getStateEncoding(self, state):
		"""
			Returns one-hot encoding of state as 1D numpy array, by appending one-hot vectors for x and y coordinates. 
		"""
		return state.getEncoding()

	def getStateListEncoding(self, state_list):
		encoding = []
		for i in range(len(state_list)):
			encoding.append(self.getStateEncoding(state_list[i]))
		return np.array(encoding, dtype=np.float32)

	def getActionEncoding(self, action):
		"""
			Returns one-hot encoding of action as 1D numpy array.
		"""
		if (self.encoding_type == 'one-hot'):
			encoding = [0.0 for _ in range(self.action_encoding_dim)]
			encoding[self.actions_to_index[action]] = 1.0
			return np.array(encoding, dtype=np.float32)
		else:
			return np.array(util.getBinaryEncoding(self.actions_to_index[action], self.action_encoding_dim), dtype=np.float32)