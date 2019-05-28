import numpy as np
import sys

def normalise(X):

	mean = np.sum(X,axis=0) / np.size(X,axis=0)
	minimum = np.amin(X,axis=0)
	maximum = np.amax(X,axis=0)

	for i in range(np.size(X,1)):
		X[:,i] = X[:,i] - mean[i]
		if maximum[i] - minimum[i] != 0:
			X[:,i] = X[:,i] / (maximum[i] - minimum[i])

	return X

if __name__ == "__main__":

	X_test = np.loadtxt(sys.argv[1], delimiter=",")
	X_test = normalise(X_test)
	X_test = np.insert(X_test, 0, 1, axis=1)

	params = np.loadtxt("params.txt", delimiter=",")

	Y_estimated = np.dot(X_test,params)

	np.savetxt(sys.argv[2], Y_estimated.astype(int), fmt='%d', delimiter='\n')

