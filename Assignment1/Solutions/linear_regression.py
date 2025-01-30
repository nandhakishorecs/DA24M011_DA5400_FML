# ------------------------------------------ Linear Regression ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 1 
#
#   This file contains code for implementing linear regression on a dataset with 
#   'n' features and 'd' dimensions using closed form solution, gradient descent, 
#   stochastic gradient descent and ridge regularisation 
#   
#   Description of functions in the class 'linear_model'
#
#       - The inputs for the classes are as follows: 
#           *   X: numpy array (matrix)
#           *   y: vector
#           *   Learning_Rate: floating point variable 
#           *   alpha: floating point variable (L2 regualisation constant)
#
#       - When called, the class is initialised with learning rate as 0.01, maximum number of iterations 
#         as 10_000 and minimum tolerance as 1e-9
#
#       - analytic_Solution(X, y)
#           *   solves the linear regression problem using analytic solution 
#           *   arg: fit_intercept: bool - includes bias in the dataset, when set to True
#           *   arg: alpha: float - the value of regularisation constant , initialised as zero by default
#
#       - fit (X, y) 
#           *   solves the linear regression problem using 'gradient descent algorithm'
#           *   arg: fit_intercept: bool - includes bias in the dataset, when set to True
#           *   arg: alpha: float - the value of regularisation constant , initialised as zero by default
#
#		- SGD (X, y)
#			*	solves the linear regression problem using 'stochastic gradient descent algorithm'
#			*	arg: fit_intercept: bool - includes bias in the dataset, when set to True
#           *   arg: alpha: float - the value of regularisation constant , initialised as zero by default
#			* 	arg: batch_size: int - total number of batches, the dataset is split into when trained
#					 set to 100 by default
#
#		- cross_validate(X, y, parameters, split_size)
#			* 	performs cross validation for the regularisation constant of ridge regression 
#			*	arg: parameters - a list of parameters, which are need to cross validated 
#			* 	arg: split_size - it is the K in K Fold cross validation - by default it does 
# 					 5 fold cross validation 
#
#       - predict(X): 
#           *   returns the predictions for given datapoint 
#           *   arg: fit_intercept: bool - includes bias in the dataset, when set to True
#               
#       - MSE & SSE functions return the Mean Squared Error and Sum of Squared Errors for the 
#         given predictions 
#
#       - parameter(): 
#           *   returns the weights and bias of the choosen model 
#           *   arg: fit_intercept: bool - includes bias in the dataset, when set to True
# --------------------------------------------------------------------------------------------------------------

# importing libraries 

# for progress bar
from tqdm.auto import tqdm		 # type: ignore

# matrix , vector operations 
import numpy as np				# type: ignore
from numpy import * 			# type: ignore

class linear_model:

	# -------------------------------------- Class Initialisation --------------------------------------------

	__slots__ = '_lr', '_iters', '_tolerance' ,'_w', '_b'
	def __init__(self, learning_rate = 0.01, iterations = 10_000, tolerance = 1e-9) -> None:
		self._lr = learning_rate
		self._iters = iterations
		self._w = None
		self._b = None
		self._tolerance = tolerance

	# ------------------------------------------ Analytic Solution -------------------------------------------

	def analytical_solution(self, X: np.ndarray, y: np.ndarray, fit_intercept: bool = False, alpha:float = 0) -> np.ndarray:
		if(fit_intercept == False):
			penalty = np.dot(alpha, np.identity(X.shape[1]))
			pseudo_inverse = np.linalg.inv(np.dot(X.T, X) + penalty)
			self._w = np.dot(np.dot(pseudo_inverse, X.T), y)
			return self._w

		elif(fit_intercept == True):
			X_bias = np.c_[X, np.ones(X.shape[0])]
			penalty = np.dot(alpha, np.identity(X_bias.shape[1]))
			pseudo_inverse = np.linalg.inv(np.dot(X_bias.T, X_bias) + penalty)
			w = np.dot(np.dot(pseudo_inverse, X_bias.T), y)

			# Book keeping for maintaining values of weights and bias in the class 
			self._w = np.empty((X.shape[1], 1))

			for i in range(X_bias.shape[1]-1):
				self._w[i] = w[i]

			self._b = w[w.shape[0]-1]
            # End of Book keeping 
			
			return w

	# -------------------------------- Helper Function to do prediction --------------------------------------

	def predict(self, X: np.ndarray, fit_intercept:bool = False) -> np.ndarray:
		if(fit_intercept == True): 
			return np.dot(X, self._w) + self._b
		if(fit_intercept == False): 
			return np.dot(X, self._w)
		
	# --------------------------------- Solution using Gradient Descent --------------------------------------
		
	def fit(self, X:np.ndarray, y:np.ndarray, fit_intercept:bool = False, alpha:float = 0):

		# Performing calculations for 
		w_MLE = self.analytical_solution(X, y, fit_intercept=fit_intercept)
		
		# To store the euclidien distance between analytic solution and gradient descent solution
		results = [] 

		# for unbaised feature(s)
		n_samples, n_features = X.shape
		
		# Expanding the features to accomodate bias 
		X_bias = np.empty((X.shape[1], 1))

		if(fit_intercept == False): 
			w_no_bias = np.random.randn(n_features,1)

		elif(fit_intercept == True): 
			X_bias = np.c_[X, np.ones(n_samples)]
			w_bias = np.random.randn(X_bias.shape[1], 1)

		# Tqdm Counter initialisation 
		tq = tqdm(range(self._iters), ncols = 99, desc = 'Iterations', disable = True)

		# Counter for loss tolerance 
		previous_loss = float('inf')

		for _ in tq:

			# Gradient descent when there is no bias term in the fit 
			if(fit_intercept == False): 
				y_pred = np.dot(X, w_no_bias)

				loss = np.sum((y_pred - y) ** 2)	
				current_loss = self.MSE(y_pred, y) 
				
				# parameter updation
				dw = (2/n_samples) * np.dot(X.T, y_pred - y) + (2 * alpha * w_no_bias)
				w_no_bias = w_no_bias - (self._lr * dw)
				
				# euclidien distance between analytic solution and gradient descent solution
				results.append(np.linalg.norm(w_no_bias - w_MLE ))

				# if all(abs(self._w - flag ) <= self._tol):
				if (abs(previous_loss - current_loss).all() <= self._tolerance):	# Efficient numpy code
					
					self._w = w_no_bias				
					
					print(f'\nStopped at Iteration: {_} !\n')
					break

				previous_loss = current_loss
				tq.set_postfix({f'Loss': loss})

			# Gradient descent when there is a bias term in the fit 
			elif(fit_intercept == True): 
				y_pred = np.dot(X_bias, w_bias)

				# parameter updation
				dw = (2/n_samples) * np.dot(X_bias.T, y_pred - y) +  (2 * alpha * w_bias)
				w_bias = w_bias - (self._lr * dw)

				loss = np.sum((y_pred - y) ** 2)	
				current_loss = self.MSE(y_pred, y)

				# euclidien distance between analytic solution and gradient descent solution
				results.append(np.linalg.norm(w_bias - w_MLE ))					

				# if all(abs(self._w - flag ) <= self._tol):
				if (abs(previous_loss - current_loss).all() <= self._tolerance):	# Efficient numpy code

					self._w = np.empty((X.shape[1], 1))
					for i in range(X_bias.shape[1]-1):
						self._w[i] = w_bias[i]

					self._b = w_bias[w_bias.shape[0]-1]

					print(f'\nStopped at Iteration: {_} !\n')
					break

				previous_loss = current_loss
				tq.set_postfix({f'Loss\t': loss})

		return results

	# ------------------------- Solution using Stochastic Gradient Descent ---------------------------------

	def SGD(self, X, y, fit_intercept = False, batch_size = 100, alpha = 0): 
		
		# Performing calculations for analytic solution
		w_MLE = self.analytical_solution(X, y, fit_intercept = fit_intercept)
		
		# To store the euclidien distance between analytic solution and stochastic gradient descent solution
		results = [] 

		# for unbaised feature(s)
		n_samples, n_features = X.shape

		# Expanding the features to accomodate bias 
		X_bias = np.empty((X.shape[1], 1))

		if(fit_intercept == False): 
			w_no_bias = np.random.randn(n_features,1)

		elif(fit_intercept == True): 
			X_bias = np.c_[X, np.ones(n_samples)]
			w_bias = np.random.randn(X_bias.shape[1], 1)

		# Tqdm Counter initialisation 
		tq = tqdm(range(self._iters), desc = 'Iterations', disable = True)

		# Counter for loss tolerance 
		previous_loss = float('inf')

		for _ in tq:

			# Gradient descent when there is no bias term in the fit 
			if(fit_intercept == False): 
				
				# Splitting dataset into batches and training the batches
				random_indices = np.random.choice(n_samples, batch_size, replace = False )

				X_batch = X[random_indices] 
				y_batch = y[random_indices]
				
				y_pred = np.dot(X_batch, w_no_bias)

				loss = np.sum((y_pred - y_batch) ** 2)	
				current_loss = self.MSE(y_pred, y_batch) 
				
				# parameter updation
				dw = (2/n_samples) * np.dot(X_batch.T, y_pred - y_batch) + (2 * alpha * w_no_bias)
				w_no_bias = w_no_bias - (self._lr * dw)
				
				# euclidien distance between analytic solution and gradient descent solution
				results.append(np.linalg.norm(w_no_bias - w_MLE ))

				if (abs(previous_loss - current_loss).all() <= self._tolerance):	# Efficient numpy code
					
					self._w = w_no_bias				
					
					print(f'Stopped at Iteration: {_} !')
					break

				previous_loss = current_loss
				tq.set_postfix({f'Loss\t': loss})

			# Gradient descent when there is a bias term in the fit 
			elif(fit_intercept == True): 

				# Splitting dataset into batches and training the batches
				random_indices = np.random.choice(n_samples, batch_size, replace = False )

				X_batch = X_bias[random_indices] 
				y_batch = y[random_indices]

				y_pred = np.dot(X_batch, w_bias)

				# parameter updation
				dw = (2/n_samples) * np.dot(X_batch.T, y_pred - y_batch) + (2 * alpha * w_bias)
				w_bias = w_bias - (self._lr * dw)

				loss = np.sum((y_pred - y_batch) ** 2)	
				current_loss = self.MSE(y_pred, y_batch)

				# euclidien distance between analytic solution and gradient descent solution
				results.append(np.linalg.norm(w_bias - w_MLE ))
				if (abs(previous_loss - current_loss).all() <= self._tolerance):	# Efficient numpy code
					
					# w_sgd = np.mean(np.array(w_sgd), axis = 0)
					self._w = np.empty((X.shape[1], 1))
					for i in range(X_bias.shape[1]-1):
						self._w[i] = w_bias[i]

					self._b = w_bias[w_bias.shape[0]-1]

					print(f'Stopped at Iteration: {_} !')
					break

				previous_loss = current_loss
				tq.set_postfix({f'Loss\t': loss})

		return results

	# ------------------------------------ Cross Validation ---------------------------------------

	def cross_validate(self, X: np.ndarray, y:np.ndarray, parameters:np.ndarray, split_size = 5):
		# Doing 5 fold cross validation by default. 

		n_samples, _ = X.shape 

		# picking random indicies for picking a batch of data of size 100
		random_indices = np.random.choice(n_samples, n_samples//split_size, replace = False)

		# validation data 
		X_val = X[random_indices]
		y_val = y[random_indices]

		# training data with parameters and validating for choosing the best 
		X_train = [] 
		y_train = [] 

		# populating X_train and y_train 
		for i in range(n_samples): 
			if i not in random_indices: 
				X_train.append(X[i])
				y_train.append(y[i])

		# converting list to numpy array for ease of computng 
		X_train = np.array(X_train)
		y_train = np.array(y_train)

		# Variable to store the min_error for the best parameter
		error = []

		# performing cross validation 
		for i in range(0, len(parameters)): 
			self.fit(X_train, y_train, alpha = parameters[i], fit_intercept=True)
			y_pred = self.predict(X_val)
			error.append(self.MSE(y_val, y_pred))				
		
		return error

	# ----------------------------------------- Loss Functions --------------------------------------------
	# Sum of Squared Error
	def SSE(self, y: np.ndarray, y_pred: np.ndarray) -> float:
		error = 0
		for i in range(0, len(y_pred)):
			error += (y[i] - y_pred[i]) ** 2
		return error

	# Mean of sum of Squared Error
	def MSE(self, y: np.ndarray, y_pred: np.ndarray) -> float:
		error = 0
		n_samples = len(y_pred)
		for i in range(0, len(y_pred)):
			error += (y[i] - y_pred[i]) ** 2
		error = error/n_samples
		return error

	# ----------------------------------- Parameters after estimation ------------------------------------

	def parameters(self, fit_intercept = False):
		if(fit_intercept == True):
			return self._w, self._b
		elif(fit_intercept == False): 
			return self._w