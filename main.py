import numpy as np
import neuralnet
import activators

# Features & Labels
features = np.array([[0, 0, 1],
					 [0, 1, 1],
					 [1, 0, 1],
					 [1, 1, 1]])

labels = np.array([[0, 1, 1, 0]]).T

if __name__ == '__main__':
	print("Final result: ", neuralnet.neuralnet(features, labels))