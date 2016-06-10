import numpy as np
np.random.seed(1)

# Activators
def logistic(x, dS = False):
	if dS: return logistic(x) * (1 - logistic(x))

	return 1 / (1 + np.exp(-x))

def tanh(x, dS = False):
	if dS: return 1 - tanh(x) ** 2
	
	return np.tanh(x)

# Features & Labels
features = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

labels = np.array([[0], [1], [1], [0]])

#synapses
synapse0 = 2 * np.random.random((3, 4)) - 1
synapse1 = 2 * np.random.random((4, 4)) - 1
synapse2 = 2 * np.random.random((4, 1)) - 1


for iteration in range(6000):

	layer0 = features
	layer1 = logistic(np.dot(layer0, synapse0))
	layer2 = logistic(np.dot(layer1, synapse1))
	layer3 = logistic(np.dot(layer2, synapse2))

	# Backpropagation
	l3_error = labels - layer3
	d_layer3 = l3_error * logistic(layer3, dS = True)
	
	if iteration % 1000 == 0: print("Error: ", np.mean(np.abs(l3_error)) * 100, "%")
	
	l2_error = d_layer3.dot(synapse2.T)
	d_layer2 = l2_error * logistic(layer2, dS = True)
	
	l1_error = d_layer2.dot(synapse1.T)
	d_layer1 = l1_error * logistic(layer1, dS = True)

	# Gradient Descent
	synapse2 += layer2.T.dot(d_layer3)
	synapse1 += layer1.T.dot(d_layer2)
	synapse0 += layer0.T.dot(d_layer1)
	
print("Final result: ", layer3)