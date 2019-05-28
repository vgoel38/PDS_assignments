import numpy as np
import sys
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss


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
	data_train = data[:int(np.size(data,0)*0.8)]
	data_val = data[int(np.size(data,0)*0.6):int(np.size(data,0)*0.8)]
	data_test = data[int(np.size(data,0)*0.8):]
	np.savetxt('train_data.txt', data_train, fmt="%s")
	np.savetxt('val_data.txt', data_val, fmt="%s")
	np.savetxt('test_data.txt', data_test, fmt="%s")
	print("Data divided..")

	#Parse data
	X_train,Y_train = data_parser(data_train)
	X_val,Y_val = data_parser(data_val)
	X_test,Y_test = data_parser(data_test)
	print("Data parsed..")

	#Fit
	clf = DecisionTreeClassifier()
	clf.fit(X_train, Y_train)
	print("Data fit..")

	#Predict
	print(hamming_loss(Y_test,clf.predict(X_test)))


