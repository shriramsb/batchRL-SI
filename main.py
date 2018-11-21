import math
from windy_gridworld import *
from windy_gridworld_agent import *

def simulateEpisodes(environment, agent, num_episodes, test_every=50):
    episode = 0
    while (episode < num_episodes):
        environment.resetCurrentState()
        print('Episode %d' % (episode, ))
        num_steps = 0
        while (True):
            state = environment.getCurrentState()
            action = agent.getAction(state, train=True)
            reward, next_state, is_terminal = environment.takeAction(action)
            if (num_steps == 15000):
                is_terminal = True
            num_steps += 1
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

            



environment = WindyGridworld(is_king_moves=False, is_stochastic_wind=False)
learning_hparams = {'learning_rate' : 0.01, 'momentum' : 0.9, 'batch_size' : 16}
agent = BatchRLAgent(ER_epochs=10, episodes_per_batch=2, epsilon=0.1, gamma=environment.gamma, learning_hparams=learning_hparams, 
                        multi_output=True, gpu_id=1)
dropout_input = 0.0
dropout_hidden = 0.0
agent.initQNetwork(environment.getCurrentState(), GridAction.all_actions, dropout_input, dropout_hidden)

num_episodes = 400
simulateEpisodes(environment, agent, num_episodes)