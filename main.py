import math
from windy_gridworld import *
from windy_gridworld_agent import *

def simulateEpisodes(environment, agent, num_episodes, test_every=50):
    for episode in range(num_episodes):
        environment.resetCurrentState()
        while (True):
            state = environemnt.getCurrentState()
            action = agent.getAction(state, train=True)
            reward, next_state, is_terminal = environemnt.takeAction(action)

            agent.update((state, action, reward, next_state), is_terminal)
            if (is_terminal):
                break
        
        if (episode % test_every == 0):
            cumulative_reward = 0.0
            decay = 1.0
            while (True):
                state = environemnt.getCurrentState()
                action = agent.getAction(state, train=False)
                reward, next_state, is_terminal = environemnt.takeAction(action)
                cumulative_reward += decay * reward
                decay *= environemnt.gamma
                if (is_terminal):
                    break

            



environemnt = WindyGridworld(is_king_moves=False, is_stochastic_wind=False)
hparams = {}
agent = BatchRLAgent(ER_iters=10, episodes_per_batch=50, batch_size=128, epsilon=0.1)

num_episodes = 400
simulateEpisodes(environemnt, agent, num_episodes)