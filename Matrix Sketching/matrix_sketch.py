import sys
import numpy as np

def zero_row_present(X):

	m = np.size(X,axis=1)
	n = np.size(X,axis=0)

	for i in range(n):
		flag=0
		for j in range(m):
			if X[i][j]!=0:
				flag=1
				break
		if(flag==0):
			return i
	return -1


def sketch_matrix(X,l):

	m = np.size(X,axis=1)
	n = np.size(X,axis=0)

	B = np.zeros((l,m))

	for i in range(n):

		print(i)

		temp = zero_row_present(B)
		if temp !=-1:
			B[temp,:] = X[i,:]
		else:
			U,S,V = np.linalg.svd(B,full_matrices=False)
			
			ind = int(np.size(S)/2)
			d = S[ind]*S[ind]

			S = np.square(S)
			S = np.subtract(S,d)
			for i in range(np.size(S)):
				S[i] = max(S[i],0)
			S = np.sqrt(S)

			I = np.identity(np.size(S))
			for i in range(np.size(S)):
				I[i][i] = S[i]

			B = np.dot(I,V)

	return B


if __name__ == "__main__":

	#Load data
	X = np.loadtxt("PDS_Data1.txt", delimiter=",")
	print("Data loaded..")

	X = X[:,1:]

	B = sketch_matrix(X,np.size(X,1)-1)

	np.savetxt('sketch_data.txt', B, delimiter=',')
