import numpy as np
import neuralnet
import activators
import csv_utils
import preprocess
import csv

def add_bias(seq):
	return list(seq) + [1]


data = csv_utils.csv_read('breeds.csv')

out_obj = preprocess.DataPreprocessor(data)
features, labels = out_obj.clear()


def csv_write(param, data, filename = 'breeds.csv'):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")
        writer.writerow(param)
        for row in data:
            writer.writerow(row)

# csv_write(("male weight", "female weight", "male height", "female height", "breed"), features, filename = "normalbreeds.csv")


if __name__ == '__main__':
	model = neuralnet.neuralnet_train(np.array(map(add_bias, features), dtype = np.float16),
						np.array(labels, dtype = np.float16),
						show_err = True, alpha = 1, hidden_layer = 8, iters = 100000, batch_size = 1)
	data = [0.758, 0.8041, 0.65, 0.766, 1]
	prediction = preprocess.vector_to_label((map(round, neuralnet.neuralnet_predict(data, model))))
	
	print(prediction)
