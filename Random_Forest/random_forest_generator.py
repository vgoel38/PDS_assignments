import sys
import numpy as np
import math
import random
import os

M = 50		#Number of bootstrap samples
m = 10		#Number of candidate features at each level
depth = 6	#tree depth
internal_nodes = dict()
leaves = dict()


def find_mean(X):
	result = 0
	for i in range(np.size(X,0)):
		result += X[i][0]
	if np.size(X,0) == 0:
		return 0
	else:
		return result/np.size(X,0)


def find_cost(selected_split_pt, selected_feature, X):
	X_left = []
	X_right = []
	for i in range(np.size(X,0)):
		if X[i][selected_feature] <= selected_split_pt:
			X_left.append(X[i])
		else:
			X_right.append(X[i])

	left_mean = find_mean(X_left)
	right_mean = find_mean(X_right)

	left_sum = 0
	for i in range(np.size(X_left,0)):
		left_sum += np.square(X_left[i][0] - left_mean)
	if np.size(X_left,0) !=0:
		left_sum = left_sum/(np.size(X_left,0))

	right_sum = 0
	for i in range(np.size(X_right,0)):
		right_sum += np.square(X_right[i][0] - right_mean)
	if np.size(X_right,0) !=0:
		right_sum = right_sum/(np.size(X_right,0))

	return left_sum + right_sum


def create_node(X, selected_features):

	cost = math.inf
	best_feature = 0
	best_split_pt = 0

	for j in selected_features:
		for i in range(np.size(X,0)):
			temp_cost = find_cost(X[i][j], j, X)
			if temp_cost < cost:
				best_feature = j
				best_split_pt = X[i][j]
				cost = temp_cost

	return best_feature, best_split_pt


def create_tree(current_depth, node, X):

	if np.size(X,0) == 0:
		return

	if current_depth == depth:
		y_mean = find_mean(X)
		leaves[node] = y_mean
		return

	#draw random subset of m features
	selected_features = dict()
	F = np.size(X,1)-1 #no. of features
	for i in range(m):
		val = random.randint(1,F)
		while val in selected_features:
			val = random.randint(1,F)
		selected_features[val] = 1

	
	#pick the best feature with best split point
	print("creating node "+str(node))
	best_feature, best_split_pt = create_node(X,selected_features)
	internal_nodes[node] = best_feature, best_split_pt, find_mean(X)

	#divide data space into X_left, X_right
	X_left = []
	X_right = []
	for i in range(np.size(X,0)):
		if X[i][best_feature] <= best_split_pt:
			X_left.append(X[i])
		else:
			X_right.append(X[i])

	create_tree(current_depth+1, node*2+1, X_left)
	create_tree(current_depth+1, node*2+2, X_right)

def print_tree(i):
	if os.path.exists("Trees/tree_"+str(i)+".txt"):
  		os.remove("Trees/tree_"+str(i)+".txt")

	for elem in internal_nodes:
		print('0',elem, internal_nodes[elem][0], internal_nodes[elem][1], internal_nodes[elem][2], file=open("Trees/tree_"+str(i)+".txt", "a"))
	for elem in leaves:
		print('1',elem, leaves[elem], '0', '0', file=open("Trees/tree_"+str(i)+".txt", "a"))


for i in range(M):

	#load bootsrap sample i
	X = np.loadtxt("BS/BS_"+str(i)+".txt", delimiter=",")
	print("Sample "+str(i)+" loaded..")
	
	internal_nodes.clear()
	leaves.clear()

	create_tree(0,0,X)
	print("Tree "+str(i)+" created..")
	
	print_tree(i)
	
