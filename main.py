import numpy as np
import neuralnet
import activators
import csv_utils
import preprocess

def add_bias(seq):
	return list(seq) + [1]

data = csv_utils.csv_read('breeds.csv')

out_obj = preprocess.DataPreprocessor(data)
features, labels = out_obj.clear()
    
if __name__ == '__main__':
	print("Final result: ",
				neuralnet.neuralnet(np.array(map(add_bias, features), dtype = np.float16),
									np.array(labels, dtype = np.float16),
									show_err = True, alpha = 0.5, hidden_layer = 32))

