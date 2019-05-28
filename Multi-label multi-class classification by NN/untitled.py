import numpy as np
import sys
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss


# def predict(X):


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


if __name__ == "__main__":

	X = np.genfromtxt("train_data_X.txt")
	Y = np.genfromtxt("train_data_Y.txt")
	X = np.append(X, np.genfromtxt("test_data_X.txt"), axis=0)
	Y = np.append(Y, np.genfromtxt("test_data_Y.txt"), axis=0)

	count = 0
	for elem in X:
		# temp = check1(elem)
		# for i in range(5):
		# 	if temp[i] == 1 and Y[count][i] == 0:
		# 		print(elem)
		# 		print(Y[count])
		temp = check2(elem)
		for i in range(5):
			# if temp[i] == 1 and Y[count][i] == 0:
			if temp[i] == 1:
				print(elem)
		count += 1
