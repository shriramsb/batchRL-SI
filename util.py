from mdp_environment import *
from mdp_batchRL_agent import *
import math

def getBinaryEncoding(num, encoding_dim):
	encoding = [0.0 for _ in range(encoding_dim)]
	for i in range(encoding_dim - 1, -1, -1):
		if (num % 2 == 1):
			encoding[i] = 1
		else:
			encoding[i] = 0
		num = math.floor(num / 2)
	return encoding
