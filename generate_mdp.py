# Blame : @divyansh
# Command : python3 generate_mdp.py n k > mdp_data/mdp_n_k.dat
'''
This generates an MDP of [0,n-1] states, with state n-1 being the terminal state.
There are (at max) k actions from each state, going to all the (<=k) successive states.
That is, there is no transition from state i to j with i >= j.

Note that higher the k, more the number of actions from each state and hence,
lesser the exploration in each episode.
'''
'''
MDP File format (copied from readme.txt) :
<number of states> = n (num of states)
<number of actions> = k (num of actions)
<Reward for (s, a) in each line, with all 'a' specified before changing 's'. ie give in each line (s_0, a_0), (s_0, a_1),... then move to (s_1, a_0). Each line should contain reward for transition to all possible next_states>
<Transition function same as reward format>
<gamma> = 1 (since episodic)
<start states separated by space> = 0 1 2 ... n-2
<terminal states separated by space> = n-1
'''

import sys

n = int(sys.argv[1])
k = int(sys.argv[2])

print(n) # states
print(k) # actions

# print reward == -1 for all transitions except the last one - state n-1 to n-1
for s in range(n):
	for a in range(k):
		for sp in range(n):
			if s == n-1 and sp == n-1:
				print(0, end=' ')
			else:
				print(-1, end=' ')
		print("")

# print transition probabilities
for s in range(n):
	for a in range(k):
		takesToState = min(s + a + 1, n-1)
		for sp in range(n):
			if sp == takesToState:
				print(1.0, end=' ')
			else:
				print(0.0, end=' ')
		print("")

print(1) # gamma
for startState in range(n-1):
	print(startState, end=' ')
print("")
print(n-1)