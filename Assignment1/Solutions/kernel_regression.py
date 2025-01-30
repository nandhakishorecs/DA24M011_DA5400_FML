# ------------------------------------------ Kernel Regression ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 1 
#
#   This file contains code for implementing kernel regression on a dataset with 
#   'n' features and 'd' dimensions using matrix operations with ridge regularisation 
#   
#   Description of functions in the class 'kernel_regression'
#
#       - The inputs for the classes are as follows: 
#			* 	kernel: name of the kernel function as a string 
#           *   alpha: floating point variable (L2 regualisation constant)
#
#       - When called, the class is initialised with a regualrisation constant as 0,
# 		  default kernel as radial basis kernel
#
#       - fit (X_train, y_train, X_test) 
#           *   computes the kernel matrix for X_train and kernel matrix using X_test and 
# 				X_train for predictions, computes the weights and returns the predictions
#           *   arg: kernel: name of the kernel function as a string 
#           *   arg: gamma - precision (reciprocal of standard deviation) value for radial basis kernel
# 		
# --------------------------------------------------------------------------------------------------------------

# Importing Libraries 

# Matrix Operations 
import numpy as np 						 # type: ignore
from numpy import *						# type: ignore

class kernel_regression:
	__slots__ = '_w', '_alpha', '_kernel'
	# weights, regularisation constant and type of kernel 

	# -------------------------------------- Class Initialisation --------------------------------------------

	def __init__(self, alpha:float = 0, kernel:str = 'rbf_kernel') -> None:
		self._w = None 
		self._alpha = alpha
		self._kernel = kernel

	# ----------------------------------- Generating kernel matrix --------------------------------------------

	def fit(self, X_train, y_train, X_test, kernel:str, gamma:float = 1):

		N_train, _ = X_train.shape
		N_test, _ = X_test.shape

		# Compute kernel matrix initialisation training and test data
		K_train = np.zeros((N_train, N_train))
		K_test = np.zeros((N_test, N_train))

		# Compute kernel matrix for training and test data
		for i in range(N_train):
			# RBF Kernel 
			if(kernel == 'rbf_kernel'):
				for j in range(N_train):
					K_train[i, j] = self.rbf_kernel(X_train[i], X_train[j], gamma)
				for j in range(N_test):
					K_test[j, i] = self.rbf_kernel(X_test[j], X_train[i], gamma)
			# Linear Kernel 		
			elif(kernel == 'linear_kernel'): 
				for j in range(N_train):
					K_train[i, j] = self.linear_kernel(X_train[i], X_train[j])
				for j in range(N_test):
					K_test[j, i] = self.linear_kernel(X_test[j], X_train[i])
			# Quadratic Kernel 			
			elif(kernel == 'quadratic_kernel'): 
				for j in range(N_train):
					K_train[i, j] = self.quadratic_kernel(X_train[i], X_train[j])
				for j in range(N_test):
					K_test[j, i] = self.quadratic_kernel(X_test[j], X_train[i])
		
		# Compute weights using regularized least squares
		weights = np.linalg.inv(K_train + self._alpha * np.identity(N_train)) @ y_train

		# Predict target values for the test set
		y_pred = K_test @ weights

		return y_pred

	# ----------------------------------- Kernel Functions ---------------------------------------------
	def linear_kernel(self, x1, x2):
		return np.dot(x1, x2)

	def quadratic_kernel(self, x1, x2): 
		return (np.dot(x1, x2) + 1) ** 2

	def rbf_kernel(self, x1, x2, gamma=1):
		return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
	
	def laplacian_kernel(self, x1, x2): 
		''' Incomplete '''
		pass