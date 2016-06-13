import numpy as np

def logistic(x, dS = False):
	if dS: return x * (1 - x)

	return 1 / (1 + np.exp(-x))

def tanh(x, dS = False):
	if dS: return 1 - x ** 2
	
	return np.tanh(x)

def relu(x, dS = False):
	if dS: return 1 if x > 0 else 0.01
	
	return max(x, 0.01 * x)