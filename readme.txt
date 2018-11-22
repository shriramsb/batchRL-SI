MDP file format:
<number of states>
<number of actions>
<Reward for (s, a) in each line, with all 'a' specified before changing 's'. ie give in each line (s_0, a_0), (s_0, a_1),... then move to (s_1, a_0). Each line should contain reward for transition to all possible next_states>
<Transition function same as reward format>
<gamma>
<start states separated by space>
<terminal states separated by space>

Look at mdp_environment::MDPEnvironment init function for more details.

In mdp_main.py, change line 40 for the mdp_file_path in above format
For lines 41-46
Line 41, Optimizer uses momentum+SGD. 
ER_epochs : 'k' in paper[1] Algorithm 2
episodes_per_batch : 'm' in paper[1] Algorithm 1
epsilon : epsilon for epsilon greedy action selection
leave multi_output as True and gpu_id=-1 if u don't have GPU
Currently not using Dropout. So, dropout_input=0.0 (Probability of dropping input), dropout_hidden=0.0 (prob. of dropping hidden neuron).
num_layers : number of layers in the network (number of hidden layers = num_layer - 1)
hidden_dim : number of hidden neurons in each layer

[1] : Batch Reinforcement Learning in a Complex Domain