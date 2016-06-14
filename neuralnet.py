# TODO adaptive alpha

import numpy as np
from activators import logistic
np.random.seed(1)



def neuralnet(features,
			  labels,
			  alpha = 0.5,
			  hidden_layer = 4,
			  iters = 50000,
			  show_err = False,
			  activation_f = logistic,
			  batch_size = 10):
	"""
		This function trains the ANN and then
		computes the labels for given features.
	"""
	
	in_layer = len(features[0])
	print(in_layer)
	out_layer = len(labels[0])
	print(out_layer)
	
	#synapses
	synapse0 = 2 * np.random.random((in_layer, hidden_layer)) - 1
	synapse1 = 2 * np.random.random((hidden_layer, out_layer)) - 1
	
	for i in range(iters):
		# mini - batch GD
		sgd_index = np.random.randint(len(features) - batch_size)
		
		layer0 = features[sgd_index:sgd_index + batch_size]
		layer1 = activation_f(np.dot(layer0, synapse0))
		layer2 = activation_f(np.dot(layer1, synapse1))
		
		# Computing the errors
		l2_error = layer2 - labels[sgd_index:sgd_index + batch_size]
		d_layer2 = l2_error * activation_f(layer2, dS = True)
		
		l1_error = d_layer2.dot(synapse1.T)
		d_layer1 = l1_error * activation_f(layer1, dS = True)
	
		# Gradient Descent aka Backpropagation
		synapse1 -= alpha * layer1.T.dot(d_layer2)
		synapse0 -= alpha * layer0.T.dot(d_layer1)
		
		if show_err:
			if i % 1000 == 0: print("Error: ", np.mean(np.abs(l2_error)) * 100, "%")
	return layer2

