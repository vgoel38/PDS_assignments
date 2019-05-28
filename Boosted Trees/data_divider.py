import numpy as np
import sys
import math


if __name__ == "__main__":

	#Load data
	X = np.loadtxt("data1.txt", delimiter=",")
	print("Data loaded..")

	#Shuffle data and divide into training, validation and test sets
	np.random.shuffle(X)
	X_train = X[:int(np.size(X,0)*0.6),:]
	X_val = X[int(np.size(X,0)*0.6):int(np.size(X,0)*0.8),:]
	X_test = X[int(np.size(X,0)*0.8):,:]

	np.savetxt('train_data.txt', X_train, delimiter=',')
	np.savetxt('val_data.txt', X_val, delimiter=',')

	X_test = X_train[:,1:]
	np.savetxt('test_data.txt', X_test, delimiter=',')
	print("Data divided..")