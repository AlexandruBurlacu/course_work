# TODO adaptive alpha and mu
import numpy as np
from activators import logistic
np.random.seed(1)
log = []


def neuralnet_train(features,
			  labels,
			  alpha = 0.5,
			  hidden_layer = 4,
			  iters = 50000,
			  show_err = False,
			  activation_f = logistic,
			  batch_size = 10, mu = 0.5):
	"""
		This function trains the ANN and then
		computes the labels for given features.
	"""
	
	# do_dropout = True
	# dropout_percent = 0.5
	in_layer = len(features[0])
	out_layer = len(labels[0])
	
	v0 = 0; v1 = 0
	
	#synapses
	synapse0 = 2 * np.random.random((in_layer, hidden_layer)) - 1
	synapse1 = 2 * np.random.random((hidden_layer, out_layer)) - 1
	
	
	for i in range(iters):
		# mini - batch GD
		sgd_index = np.random.randint(len(features) - batch_size)
		
		layer0 = features[sgd_index:sgd_index + batch_size]
		layer1 = activation_f(np.dot(layer0, synapse0))
		# if do_dropout:
		# 	layer1 *= np.random.binomial([np.ones((len(layer0),hidden_layer))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
		layer2 = activation_f(np.dot(layer1, synapse1))
		
		# Computing the errors
		l2_error = layer2 - labels[sgd_index:sgd_index + batch_size]
		d_layer2 = l2_error * activation_f(layer2, dS = True)
		
		l1_error = d_layer2.dot(synapse1.T)
		d_layer1 = l1_error * activation_f(layer1, dS = True)
		
		# Gradient Descent + Momentum
		v1 = mu * v1 - alpha * layer1.T.dot(d_layer2)
		synapse1 += v1
		
		v0 = mu * v0 - alpha * layer0.T.dot(d_layer1)
		synapse0 += v0
		
		err = np.mean(np.abs(l2_error))
		if show_err:
			if i % 1000 == 0: log.append(err * 100)
		
	return synapse0, synapse1, activation_f

def neuralnet_predict(features, model):
	"""
		Given a trained model (synapses and the activation function)
		and features it returns the estimated label.
	"""
	synapse0, synapse1, activation_f = model
	layer0 = features
	layer1 = activation_f(np.dot(layer0, synapse0))
	layer2 = activation_f(np.dot(layer1, synapse1))
	print log
	return layer2
	