from environment import *

# Implementation of Windy Gridworld problem
class WindyGridworld(Environment):
	def __init__(self, is_king_moves=False, is_stochastic_wind=False):
		super().__init__()

		self.is_king_moves = is_king_moves
		self.is_stochastic_wind = is_stochastic_wind

		# x-axis - horizontal-right, y-axis - vertical-up, origin at bottom-left
		self.grid_size = (10, 7)

		GridPositionState.initMembers((0, 0), tuple([v - 1 for v in self.grid_size]))
		GridAction.initMembers(is_king_moves)
		# number of states and actions
		self.num_states = self.grid_size[0] * self.grid_size[1]
		self.num_actions = 8 										# 4/8 moves specified to the agent, MDP supports more general 8 moves

		# start and terminal states
		self.start_states = [(0, 3)]
		self.terminal_states = [(7, 3)]

		self.current_state = np.random.choice(self.start_states)

		# self.wind[s] gives wind strength at state 's'
		self.wind = {}
		self.wind_x = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
		for state in itertools.product(range(self.grid_size[0]), range(self.grid_size[1])):
			self.wind[state] = (0, self.wind_x[state[0]])

		# MDP's \gamma
		self.gamma = 1.0

		# transition and reward functions
		self.T = {}
		self.R = {}
		
		# helps avoid usage of 'self' before T and R
		T = self.T
		R = self.R

		# states - positions (x, y)
		# actions - (x-direction displacement, y-direction displacement)

		# populating T, R
		# R - (-1) for all transitions
		for state in itertools.product(range(self.grid_size[0]), range(self.grid_size[1])):
			T[state] = {}
			R[state] = {}
			for action in itertools.product(range(-1, 1 + 1), range(-1, 1 + 1)):
				if (action == (0, 0)): 				# no movement action not allowed
					continue
				T[state][action] = []
				R[state][action] = []
				
				# if no stochastic wind, only one next state possible
				if (not self.is_stochastic_wind):
					next_states = [None]
					next_states[0] = tuple([i + j + k for i, j, k in zip(state, action, self.wind[state])])
					next_states[0] = GridPositionState.correctOOGridPos(next_states[0])
				# if stachastic wind, 3 possible next states
				else:
					next_states = [None for _ in range(3)]
					next_states[0] = tuple([i + j + k for i, j, k in zip(state, action, self.wind[state])])
					next_states[1] = (next_states[0][0], next_states[0][1] + 1)
					next_states[2] = (next_states[0][0], next_states[0][1] - 1)
				
				# update T based on possible next states
				for i in range(len(next_states)):
					next_states[i] = GridPositionState.correctOOGridPos(next_states[i])
					R[state][action].append((next_states[i], -1))
					T[state][action].append((next_states[i], 1 / len(next_states)))

				R[state][action] = np.array(R[state][action])
				T[state][action] = np.array(T[state][action])		
	
	def takeAction(self, action):
		state = self.current_state
		if (state in self.terminal_states):
			print("Panic!! Didn't reset after end of episode")
			sys.exit(0)

		action = GridAction.actionToDirection(action)
		index = np.random.choice(range(self.T[state][action].shape[0]), p=self.T[state][action][: , 1].astype(np.float64))
		next_state = self.T[state][action][index, 0]
		reward = self.R[state][action][index, 1]
		is_terminal = next_state in self.terminal_states
		self.current_state = next_state

		return reward, GridPositionState(next_state), is_terminal

	def resetCurrentState(self):
		if (self.current_state in self.terminal_states or self.current_state in self.start_states):
			self.current_state = np.random.choice(self.start_states)
		else:
			print("Panic!! Trying to reset in the middle of an episode")
			sys.exit(0)

	def getCurrentState(self):
		return GridPositionState(self.current_state)

class GridPositionState(State):
	"""
	Holds 2D position
	_origin - origin of grid
	_max_pos - opposite corner to origin
	"""

	def __init__(self, pos):
		self.pos = pos
	
	@classmethod
	def initMembers(cls, origin, max_pos):
		cls.origin = origin
		cls.max_pos = max_pos

	@classmethod
	def correctOOGridPos(cls, pos):
		corrected_pos = list(pos)
		for i in range(len(corrected_pos)):
			corrected_pos[i] = max(cls.origin[i], corrected_pos[i])
			corrected_pos[i] = min(cls.max_pos[i], corrected_pos[i])
		return tuple(corrected_pos)

	def getLegalActions(self):
		return GridAction.all_actions
			
class GridAction(Action):
	"""
	Holds action
	"""
	N = 'N'
	S  = 'S'
	E = 'E'
	W = 'W'
	NE = 'NE'
	NW = 'NW'
	SE = 'SE'
	SW = 'SW'
	NIL = 'NIL'
	
	action_to_direction = {N : (0, 1), 
                            S : (0, -1), 
                            E : (1, 0), 
                            W : (-1, 0),
                            NE : (1, 1), 
                            NW : (-1, 1), 
                            SE : (1, -1), 
                            SW : (-1 , -1), 
                            NIL : (0, 0)}

	@classmethod
	def initMembers(cls, is_king_moves):        
		cls.all_actions = [cls.N, cls.S, cls.E, cls.W]
		if is_king_moves:
			cls.all_actions = cls.all_actions + [cls.NE, cls.NW, cls.SE, cls.SW]
    
	@classmethod
	def actionToDirection(cls, action):
		return cls.action_to_direction[action]