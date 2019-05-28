import numpy as np
import sys
import math

def normalise(X):

	mean = np.sum(X,axis=0) / np.size(X,axis=0)
	minimum = np.amin(X,axis=0)
	maximum = np.amax(X,axis=0)

	for i in range(1,np.size(X,1)):
		X[:,i] = X[:,i] - mean[i]
		if maximum[i] - minimum[i] != 0:
			X[:,i] = X[:,i] / (maximum[i] - minimum[i])

	return X


if __name__ == "__main__":

	#Load data
	X = np.loadtxt("PDS_Data1.txt", delimiter=",")
	print("Data loaded..")

	X = normalise(X)
	print("Data normalised..")

	#Shuffle data and divide into training, validation and test sets
	np.random.shuffle(X)
	X_train = X[:int(np.size(X,0)*0.6),:]
	X_val = X[int(np.size(X,0)*0.6):int(np.size(X,0)*0.8),:]
	X_test = X[int(np.size(X,0)*0.8):,:]

	np.savetxt('train_data.txt', X_train, delimiter=',')
	np.savetxt('val_data.txt', X_val, delimiter=',')
	np.savetxt('test_data.txt', X_test, delimiter=',')
	print("Data divided..")