from environment import *
import numpy as np
import math
import util

class MDPEnvironment(Environment):
	"""
		Required parameters : 
		self.gamma 
	"""

	def __init__(self, mdp_file_path):
		"""
			Reads MDP from a file. Refer readme.txt for format of file
		"""

		super().__init__()
		f = open(mdp_file_path, "r")
		num_states = int(f.readline().split()[0])
		num_actions = int(f.readline().split()[0])

		# initialize State, Action classes with num_states and num_actions ; sets the state, action encoding dimensions
		MDPState.initMembers(num_states)
		MDPAction.initMembers(num_actions)

		self.num_states = num_states
		self.num_actions = num_actions

		# Populating T, R
		self.T, self.R = [None] * num_states, [None] * num_states
		T, R = self.T, self.R
		for s in range(num_states):
			T[s], R[s] = [None] * num_actions, [None] * num_actions
			for a in range(num_actions):
				T[s][a], R[s][a] = [None] * num_states, [None] * num_states

		for s in range(num_states):
			for a in range(num_actions):
				line_split = f.readline().split()
				for s_prime in range(num_states):
					R[s][a][s_prime] = float(line_split[s_prime])

		for s in range(num_states):
			for a in range(num_actions):
				line_split = f.readline().split()
				for s_prime in range(num_states):
					T[s][a][s_prime] = float(line_split[s_prime])
		

		self.gamma = float(f.readline().split()[0])
		
		line_split = f.readline().split()
		self.start_states = [float(v) for v in line_split]
		line_split = f.readline().split()
		self.terminal_states = [float(v) for v in line_split]

		self.current_state = int(np.random.choice(self.start_states))

	def getCurrentState(self):
		is_terminal = self.current_state in self.terminal_states
		return MDPState(self.current_state, is_terminal)

	def takeAction(self, action):
		"""
			Takes action on the current state and moves to the next state according to the transition function.
			######Returns is_terminal_before to be True only when current_state is in terminal_states, and not when next state is in terminal state
		"""
		# is_terminal_before = self.current_state in self.terminal_states
		# if (is_terminal_before):
		# 	return 0, MDPState(self.current_state, is_terminal_before), is_terminal_before
		action = MDPAction.actionToIndex(action)
		next_state = int(np.random.choice(range(self.num_states), p=self.T[self.current_state][action]))
		reward = self.R[self.current_state][action][next_state]
		self.current_state = next_state
		is_terminal = self.current_state in self.terminal_states
		return reward, MDPState(next_state, is_terminal), is_terminal
	
	def resetCurrentState(self):
		if (self.current_state not in self.terminal_states and self.current_state not in self.start_states):
			print("Panic!! Trying to reset in middle of an episode")
		self.current_state = int(np.random.choice(self.start_states))


class MDPState(object):
	"""
		Required class variables : 
		cls.encoding_dim - dimensions of 1D encoded vector
	"""

	def __init__(self, num, is_terminal=False):
		super().__init__()
		self.num = num
		self.is_terminal = is_terminal

	@classmethod
	def initMembers(cls, num_states):
		cls.num_states = num_states
		cls.encoding_dim = None

	@classmethod
	def initEncoder(cls, encoding_type):
		cls.encoding_type = encoding_type
		if (cls.encoding_type == 'one-hot'):
			cls.encoding_dim = cls.num_states
		elif (cls.encoding_type == 'binary'):
			cls.encoding_dim = math.ceil(math.log(cls.num_states, 2))

	def getEncoding(self):
		if (self.encoding_type == 'one-hot'):
			encoding = np.zeros((self.encoding_dim, ), dtype=np.float32)
			encoding[self.num] = 1.0
		elif (self.encoding_type == 'binary'):
			 encoding = np.array(util.getBinaryEncoding(self.num, self.encoding_dim), dtype=np.float32)
		return encoding

	def getLegalActions(self):
		return MDPAction.all_actions

class MDPAction(object):
	"""
		Required class variables : 
		cls.all_actions - list of all actions
	"""
	

	@classmethod
	def initMembers(cls, num_actions):
		cls.all_actions = list(range(num_actions))

	@classmethod
	def actionToIndex(cls, action):
		return action