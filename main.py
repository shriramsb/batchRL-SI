import math
from windy_gridworld import *
from windy_gridworld_agent import *

def simulateEpisodes(environment, agent, num_episodes, test_every=50):
    for episode in range(num_episodes):
        environment.resetCurrentState()
        print('Episode %d' % (episode, ))
        while (True):
            state = environment.getCurrentState()
            action = agent.getAction(state, train=True)
            reward, next_state, is_terminal = environment.takeAction(action)

            agent.update((state, action, reward, next_state), is_terminal)
            if (is_terminal):
                break
        
        if (episode % test_every == 0):
            cumulative_reward = 0.0
            decay = 1.0
            while (True):
                state = environment.getCurrentState()
                action = agent.getAction(state, train=False)
                reward, next_state, is_terminal = environment.takeAction(action)
                cumulative_reward += decay * reward
                decay *= environment.gamma
                if (is_terminal):
                    break
            
            print('testing result %f' % (cumulative_reward, ))

            



environment = WindyGridworld(is_king_moves=False, is_stochastic_wind=False)
learning_hparams = {'learning_rate' : 0.01, 'momentum' : 0.9, 'batch_size' : 16}
agent = BatchRLAgent(ER_epochs=10, episodes_per_batch=50, epsilon=0.1, gamma=environment.gamma, learning_hparams=learning_hparams)

num_episodes = 400
simulateEpisodes(environment, agent, num_episodes)