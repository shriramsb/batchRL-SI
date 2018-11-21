import itertools
import numpy as np
import sys

class Environment(object):
	def __init__(self):
		pass

	def getCurrentState(self):
		raise NotImplementedError
	
	def getValidActions(self, state):
		raise NotImplementedError

	def takeAction(self, action):
		raise NotImplementedError
	
	def resetCurrentState(self):
		raise NotImplementedError

	def isTerminal(self):
		raise NotImplementedError

class State(object):
	def __init__(self):
		self.is_terminal = False
		pass

	def getEncoding(self):
		raise NotImplementedError

class Action(object):
	def __init__(self):
		pass

	def getEncoding(self):
		raise NotImplementedError