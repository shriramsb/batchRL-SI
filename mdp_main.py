import math
from mdp_environment import *
from mdp_batchRL_agent import *
import numpy as np

def simulateEpisodes(environment, agent, num_episodes, test_every=50):
    episode = 0
    while (episode < num_episodes):
        environment.resetCurrentState()
        print('Episode %d' % (episode, ))
        while (True):
            state = environment.getCurrentState()
            action = agent.getAction(state, train=True)
            reward, next_state, is_terminal = environment.takeAction(action)
            agent.update((state, action, reward, next_state), is_terminal)
            if (is_terminal):
                break

        episode += 1
        
        if (episode % test_every == 0):
            environment.resetCurrentState()
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

            
np.random.seed(0)


environment = MDPEnvironment(mdp_file_path='mdp_data/mdp_100_25.dat')
learning_hparams = {'learning_rate' : 0.125, 'momentum' : 0.9, 'batch_size' : 16, 'use_regularisation': "True"}
agent = BatchRLAgent(ER_epochs=50, episodes_per_batch=1, epsilon=0.1, gamma=environment.gamma, learning_hparams=learning_hparams, 
                        multi_output=True, gpu_id=-1)
dropout_input = 0.0
dropout_hidden = 0.0
agent.initQNetwork(environment.getCurrentState(), MDPAction.all_actions, dropout_input, dropout_hidden, num_layers=2, hidden_dim=10)

num_episodes = 1000
simulateEpisodes(environment, agent, num_episodes)