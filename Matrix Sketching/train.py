import numpy as np
from scipy.optimize import minimize
import sys

def costFunction(params, X, Y, lamda):

	m = np.size(Y)

	J = np.dot(X,params) - Y
	J = np.dot(J,J)
	J = J/(2*m)

	Jreg = np.dot(params[1:],params[1:])
	Jreg = (Jreg*lamda)/(2*m)

	J = J + Jreg

	# grad(1,1) = sum((X*theta - y) .* X(:,1))/m;

	# X_reduced = X(:,2:size(X,2));
	# theta_reduced = theta(2:size(theta,1),:);
	# grad(2:size(grad,1),:) = (X_reduced' * (X*theta - y) + lambda*theta_reduced)/m;

	return J


if __name__ == "__main__":

	X_train = np.loadtxt("sketch_data.txt", delimiter=",")
	print("Training Data loaded...")
	Y_train = X_train[:,[0]]
	Y_train = Y_train.flatten()
	X_train = X_train[:,1:]
	X_train = np.insert(X_train, 0, 1, axis=1)

	# X_val = np.loadtxt("val_data.txt", delimiter=",")
	# print("Validation data loaded...")
	# Y_val = X_val[:,[0]]
	# Y_val = Y_val.flatten()
	# X_val = X_val[:,1:]
	# X_val = np.insert(X_val, 0, 1, axis=1)

	# lamda = 0.001

	# while lamda<10 :

	# 	print("Traning begins.. lamda = "+str(lamda))
	# 	params = np.ones(np.size(X_train,1))
	# 	result = minimize(costFunction, params, jac=False, args=(X_train,Y_train,lamda), options={'maxiter':200})
	# 	print(result.x)

	# 	val_error = costFunction(result.x, X_val, Y_val, 0)
	# 	print("Validation error = "+str(val_error))

	# 	lamda = lamda*2

	lamda = 0.001

	print("Traning begins.. lamda = "+str(lamda))
	params = np.ones(np.size(X_train,1))
	result = minimize(costFunction, params, jac=False, args=(X_train,Y_train,lamda), options={'maxiter':200})
	
	np.savetxt('params.txt', result.x, delimiter=',')

	# val_error = costFunction(result.x, X_val, Y_val, 0)
	# print("Validation error = "+str(val_error))

	




