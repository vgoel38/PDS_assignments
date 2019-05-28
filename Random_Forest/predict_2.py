import numpy as np
import sys
import math

M = 50 #No. of trees

def traverse_tree(tree, data_point):
	node = 0
	while tree[node][0]!=1:
		if data_point[int(tree[node][1])-1] <= tree[node][2]:
			if node*2+1 not in tree:
				return tree[node][3]
			node = node*2+1
		else:
			if node*2+2 not in tree:
				return tree[node][3]
			node = node*2+2
	return tree[node][1]



if __name__ == "__main__":

	#Loading data
	X_test = np.loadtxt(sys.argv[1], delimiter=",")

	forest = []

	#Loading forest
	for i in range(M):
		tree = dict()
		temp = np.loadtxt("Trees/tree_"+str(i)+".txt", delimiter=" ")
		for j in range(np.size(temp,0)):
			if temp[j][0] == 0:
				tree[temp[j][1]] = temp[j][0], temp[j][2], temp[j][3], temp[j][4]
			else:
				tree[temp[j][1]] = temp[j][0], temp[j][2]
		forest.append(tree)

	error = 0
	Y_estimated = []

	for i in range(np.size(X_test,0)):
		y = 0
		for tree in forest:
			y += traverse_tree(tree,X_test[i])
		y = y/M
		Y_estimated.append(y)

	Y_estimated = [ int(x) for x in Y_estimated ]

	np.savetxt(sys.argv[2], Y_estimated, fmt='%d', delimiter='\n')
