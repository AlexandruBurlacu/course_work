import numpy as np
import neuralnet
import activators
import csv_utils
import preprocess

import random

# Features & Labels
features = np.array([[0, 0, 1],
					 [0, 1, 1],
					 [1, 0, 1],
					 [1, 1, 1]])

labels = np.array([[0, 1, 1, 0]]).T

data = csv_utils.csv_read('breeds.csv')

out_obj = preprocess.DataPreprocessor(data)
features, labels = out_obj.clear()
    
if __name__ == '__main__':
	print("Final result: ", neuralnet.neuralnet(np.array(features)[::200], np.array(labels)[::200]))
