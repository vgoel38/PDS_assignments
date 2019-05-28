import numpy as np
import sys
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss, f1_score
import pickle

def check1(elem):
	Y = [0,0,0,0,0]

	for i in range(4):
		j = 8
		flag = 0
		while j<=14:
			if elem[i*13+j-2] == 0:
				flag = 1
				break
			j+=1
		if flag==0:
			Y[i+1]=1

	return Y


def check2(elem):
	Y = [0,0,0,0,0]

	if elem[12] == 0 or elem[25] == 0 or elem[38] == 0 or elem[51] == 0:
		return Y

	for i in range(4):
		j = 9
		flag = 0
		while j<=14:
			if elem[i*13+j-2] == 0:
				flag = 1
				break
			j+=1
		if flag==0:
			Y[i+1]=1

	return Y


def data_parser(data):
	X = np.zeros((np.size(data,0), 52))

	i = 0
	for point in data :
		count = 0
		for elem in point:
			if elem == '.':
				count+=1
			else:
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

	return X.astype(int)


if __name__ == "__main__":

	X_test = np.genfromtxt(sys.argv[1], dtype='str')
	# Y_test = np.genfromtxt("test_data_Y.txt")

	X_test = data_parser(X_test)

	with open('weights.pkl', 'rb') as file:
		clf = pickle.load(file)

	Y_predict = clf.predict(X_test)

	count = 0
	for elem in X_test:
		temp = check1(elem)
		for i in range(5):
			if temp[i] == 1:
				Y_predict[count][i] = 1
		temp = check2(elem)
		for i in range(5):
			if temp[i] == 1:
				Y_predict[count][i] = 1
		count += 1

	np.savetxt(sys.argv[2], Y_predict, fmt='%d', delimiter=',')



	# print(hamming_loss(Y_test,Y_predict))
	# print(f1_score(Y_test,Y_predict,average='macro'))
	# print(clf.score(X_test,Y_test))

