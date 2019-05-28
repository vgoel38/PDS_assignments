import sys
import numpy as np
import math
import random
import os
from predict import traverse_tree

M = 50		#Number of trees
m = 10		#Number of candidate features at each level
depth = 5	#tree depth
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


def update_outputs(X,tree_str):

	tree = dict()
	temp = np.loadtxt(tree_str, delimiter=" ")
	for j in range(np.size(temp,0)):
		if temp[j][0] == 0:
			tree[temp[j][1]] = temp[j][0], temp[j][2], temp[j][3], temp[j][4]
		else:
			tree[temp[j][1]] = temp[j][0], temp[j][2]

	for i in range(np.size(X,0)):
		current_pred = traverse_tree(tree,X[i])
		pred_sum[i] += current_pred
		X[i][0] = Y[i] - pred_sum[i]


X = np.loadtxt("train_data.txt", delimiter=",")
print("Traning Data loaded..")

X = X[:1000]

Y = X[:,[0]]
Y = Y.flatten()

pred_sum = np.zeros(np.size(Y))

for i in range(M):

	internal_nodes.clear()
	leaves.clear()

	create_tree(0,0,X)
	print("Tree "+str(i)+" created..")
	print_tree(i)
	update_outputs(X,"Trees/tree_"+str(i)+".txt")
	
