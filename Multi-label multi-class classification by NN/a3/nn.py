import numpy as np
import sys
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss
import pickle


def data_parser(data):
	X = np.zeros((np.size(data,0), 52))
	Y = np.zeros((np.size(data,0), 5))

	i = 0
	for point in data :
		count = 0
		j=0
		for elem in point:
			if elem == '.':
				count+=1
			elif elem == '0' or elem == '1':
				Y[i][j] = elem
				j+=int(1)
			elif elem != '"' and elem != ',':
				value = 0
				if elem == 'A':
					value = 12
				elif elem == 'K':
					value = 11
				elif elem == 'Q':
					value = 10
				elif elem == 'J':
					value = 9
				elif elem == 'T':
					value = 8
				else :
					value = ord(elem) - 50
				X[i][13*count+value] = 1
		i+=1

	return X.astype(int), Y.astype(int)



if __name__ == "__main__":

	#Load data
	data = np.genfromtxt("a3_original.csv", dtype='str')
	print("Data loaded..")

	#Divide data
	np.random.shuffle(data)
	data_train = data[:int(np.size(data,0)*0.7)]
	data_test = data[int(np.size(data,0)*0.3):]
	print("Data divided..")

	#Parse data
	X_train,Y_train = data_parser(data_train)
	X_test,Y_test = data_parser(data_test)
	# np.savetxt('train_data_X.txt', X_train, fmt="%s")
	# np.savetxt('train_data_Y.txt', Y_train, fmt="%s")
	np.savetxt('test_data_X.txt', X_test, fmt="%s")
	np.savetxt('test_data_Y.txt', Y_test, fmt="%s")
	print("Data parsed..")

	#Fit
	# clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-4, max_iter=500, random_state=None, hidden_layer_sizes=(100,), early_stopping=True, validation_fraction=0.3, verbose=True)
	clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-4, max_iter=500, random_state=None, hidden_layer_sizes=(100,))
	clf.fit(X_train, Y_train)
	with open('weights.pkl', 'wb') as file :
		pickle.dump(clf, file)
	print("Data fit..")


