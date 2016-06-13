import numpy as np
np.random.seed(1)

# Activators
def logistic(x, dS = False):
	if dS: return x * (1 - x)

	return 1 / (1 + np.exp(-x))

def tanh(x, dS = False):
	if dS: return 1 - x ** 2
	
	return np.tanh(x)

def relu(x, dS = False):
	if dS: return 1 if x > 0 else 0.01
	
	return max(x, 0.01 * x)

# Features & Labels
features = np.array([[0, 0, 1],
					 [0, 1, 1],
					 [1, 0, 1],
					 [1, 1, 1]])

labels = np.array([[0, 1, 1, 0]]).T

nodes = 4
alpha = 0.5

#synapses
synapse0 = 2 * np.random.random((3, nodes)) - 1
synapse1 = 2 * np.random.random((nodes, 1)) - 1


for iteration in range(10000):

	layer0 = features
	layer1 = logistic(np.dot(layer0, synapse0))
	layer2 = logistic(np.dot(layer1, synapse1))

	# Backpropagation
	l2_error = layer2 - labels
	d_layer2 = l2_error * logistic(layer2, dS = True)
	
	l1_error = d_layer2.dot(synapse1.T)
	d_layer1 = l1_error * logistic(layer1, dS = True)

	# Gradient Descent
	synapse1 -= alpha * layer1.T.dot(d_layer2)
	synapse0 -= alpha * layer0.T.dot(d_layer1)
	
	if iteration % 1000 == 0: print("Error: ", np.mean(np.abs(l2_error)) * 100, "%")
	
print("Final result: ", layer2)