import numpy as np
import sys
import math
import random


if __name__ == "__main__":

	#Load data
	X = np.loadtxt("train_data.txt", delimiter=",")
	print("Training Data loaded..")

	#Number of bootstrap samples
	M = 50 
	N = np.size(X,0)

	for i in range(M):
		np.random.shuffle(X)
		BS = []
		# array = dict()
		# for j in range(N):
		# 	val = random.randint(0,N-1)
		# 	if val not in array:
		# 		array[val]=0
		# 		BS.append(X[val])
		# np.savetxt('BS_'+str(i)+'.txt', BS, delimiter=',')

		for j in range(1000):
			BS.append(X[j])
		np.savetxt('BS/BS_'+str(i)+'.txt', BS, delimiter=',')

		print("Created Bootstrap sample "+str(i))