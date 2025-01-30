# ------------------------------------------ Logistic Regression ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 2
#
#   This file contains code for implementing Logistic Regression for classification on a text dataset with 
#   string as datapoint and string 'Spam' (or) 'Ham' as label. 
#   
#   Description of functions in the class 'NaiveBayesClassifier'
#
#       - When called, the class is initialised with a learning rate of 0.01 and the gradient descent  
#         algorithm with 1000 iterations 
#
#       - The inputs for the classes are as follows: 
#           *   X: sparse matrix (scipy.sparse.csr_matrix)
#           *   y: encoded vector (labels encoded as 1 or 0)
#
#       - fit (X, y) 
#           *   solves a binary classification problem using Logistic Regression, solved using gradient 
#               descent algorithm. 
#
#       - predict(X): 
#           *   returns the predictions for given datapoint 
#           *   The output is either 1 (or) 0, where 1 means that the datapoint is a spam email and 0 means
#               that the datapoint is a Ham email. 
#       
#       - _sigmoid(z): 
#           *   return the sigmoid function for a numpy array
#
#       - MSE(y_true, y_predicted): 
#           *   Returns the Mean Squared Error
# --------------------------------------------------------------------------------------------------------------

# --------------------------------------- Importing Libraries --------------------------------------
import numpy as np
from scipy.sparse import csr_matrix, issparse

class LogisticRegression:
    # --------------------------------------- Class Initialisation --------------------------------------
    __slots__ = '_learning_rate', '_n_iterations', '_weights', '_bias', '_tolerance'
    def __init__(self, learning_rate=0.01, num_iterations=1000) -> None:
        self._learning_rate = learning_rate
        self._n_iterations = num_iterations
        self._weights = None
        self._bias = None
        self._tolerance = 1e-9
    
    # --------------------------------------- Sigmoid Function --------------------------------------
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    # ------------------------------- Fit Method to train the model ---------------------------------
    def fit(self, X: csr_matrix, y: np.ndarray) -> None:
        if not issparse(X):
            raise ValueError("Input X must be a sparse matrix")
        if not isinstance(y, np.ndarray):
            raise ValueError("Input y must be a numpy array")
        
		# Initialize parameters
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0
        
        previous_loss = float('inf')
        
        for _ in range(self._n_iterations):
            # Linear model
            linear_model = X.dot(self._weights) + self._bias
            
            # Sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Loss Counter 
            current_loss = self.MSE(y_predicted, y) 
            
            # Solving for weights and bias using gradient descent 
            dw = (1 / n_samples) * X.T.dot(y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self._weights -= self._learning_rate * dw
            self._bias -= self._learning_rate * db

            if (abs(previous_loss - current_loss).all() <= self._tolerance):	# Efficient numpy code
                print(f'\nStopped at Iteration: {_} !\n')
                break
            
    # ------------------------ Helper Function to do a prection from the saved model ----------------------
    def predict(self, X: csr_matrix) -> np.ndarray:
        linear_model = X.dot(self._weights) + self._bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)
    
    # --------------------------------------- Loss Function --------------------------------------
    # Mean of sum of Squared Error
    def MSE(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        error = 0
        n_samples = len(y_pred)
        for i in range(0, len(y_pred)):
            error += (y[i] - y_pred[i]) ** 2
        error = error/n_samples
        return error