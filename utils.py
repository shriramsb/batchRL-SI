import torch
import torch.nn as nn
import torch.nn.functional as F


psi = 0.00000001 #small value to avoid division with zeros

def get_zero_like(params):
	#returns a constant tensor
	result = []
	for param in params:
		result.append(param.new_zeros(param.size()))
	return result



def get_params_data(params):
	#reutrn list of constants for corresponding list of trainable vars
	params_data = []
	for param in params:
		# WEAK : check if this is equivalent to creating a new tensor and copying the values of param.data to it
		params_data.append(param.detach().clone())

	return params_data

def get_grads_from_params(params):
	#params is list of trainable parameters

	# CHECK : Added .detach().clone() just to be safe
	gradients = []
	for param in params:
		gradients.append(param.grad.detach().clone())

	#print("params: {}".format(params))
	#print("gradients: {}".format(gradients))

	return gradients

def delta_param_gradient_product(delta_params, grads):
	# negative of delta_params * gradients
	result = []
	with torch.no_grad():
		for i in range(len(delta_params)):
			result.append(delta_params[i]*grads[i]*-1)

	'''
	print("delta_params: {}".format(delta_params))
	print("grads: {}".format(grads))
	print("result: {}".format(result))
	'''

	return result

def add_tensor_lists(list_a, list_b):
	#sum up tensors in two lists
	result = []
	for i in range(len(list_a)):
		result.append(list_a[i]+list_b[i])

	return result

def sub_tensor_lists(list_a, list_b):
	result = []
	# Change : ensure no graph construction
	with torch.no_grad():
		for i in range(len(list_a)):
			result.append(list_a[i] - list_b[i])

	return result

def get_regularisation_penalty(params, initial_params, importance):
	#returns a scalar
	#initial params must be list of constant tensors
	#params must be a list of variable tensors
	'''
	if not params:
		print("params")
	if not initial_params:
		print("initial_params")
	if not importance:
		print("importance")
	'''

	penalty = 0

	for i in range(len(params)):
		weighted_delta = importance[i] * (params[i]-initial_params[i]) ** 2
		penalty += torch.sum(weighted_delta)

	return penalty

def update_importance(previous_importance, path_integral, delta_params, importance_retain_factor):
	with torch.no_grad():
		for i in range(len(previous_importance)):
			previous_importance[i].mul_(importance_retain_factor)
			previous_importance[i].add_(path_integral[i]/(delta_params[i]**2 + psi))

	#print(previous_importance)

def add_to_tensor_lists(dest, src):
	with torch.no_grad():
		for i in range(len(dest)):
			dest[i].add_(src[i])