# ------------------------------------------ Naive Bayes Classifier ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 2
#
#   This file contains code for implementing Naive Bayes algorithm for classification on a text dataset with 
#   string as datapoint and string 'Spam' (or) 'Ham' as label. 
#   
#   Description of functions in the class 'NaiveBayesClassifier'
#
#       - The inputs for the classes are as follows: 
#           *   X: sparse matrix (scipy.sparse.csr_matrix)
#           *   y: encoded vector (labels encoded as 1 or 0)
#
#       - When called, the class is initialised with class likelihood and priors as None values.
#
#       - fit (X, y) 
#           *   solves a binary classification problem by calculating posterior distribution with likelihood
#               prior values 
#
#       - predict(X): 
#           *   returns the predictions for given datapoint 
#           *   The output is either 1 (or) 0, where 1 means that the datapoint is a spam email and 0 means
#               that the datapoint is a Ham email. 
# --------------------------------------------------------------------------------------------------------------

# ------------------------------------------- Importing Libraries ----------------------------------------
import scipy
from scipy.sparse import issparse
import numpy as np
import scipy.sparse

class NaiveBayesClassifier:
    # --------------------------------------- Class Initialisation ------------------------------------
    __slots__ = '_class_priors', '_feature_probabilities', '_classes'
    def __init__(self) -> None:
        self._class_priors = None
        self._feature_probabilities = None
        self._classes = None

    # ------------------------------- Fit method to train the model ------------------------------------
    def fit(self, X: scipy.sparse.csr_matrix, y: np.ndarray) -> None:
        # Ensure X is sparse and y is a numpy array
        if not issparse(X):
            raise ValueError("Input X must be a sparse matrix")
        if not isinstance(y, np.ndarray):
            raise ValueError("Input y must be a numpy array")

        # Identify classes and calculate prior probabilities
        self._classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self._classes)
        n_samples, n_features = X.shape

        # Calculate class prior probabilities
        self._class_priors = class_counts / y.size

        # Initialize the conditional probabilities matrix
        self._feature_probabilities = np.zeros((n_classes, n_features))

        # Calculate conditional probabilities
        for index, _classes in enumerate(self._classes):
            # Select the rows in X that correspond to the class `cls`
            class_rows = X[y == _classes]
            # Sum features per class to get word counts
            feature_counts = np.array(class_rows.sum(axis=0)).flatten()
            # Calculate the conditional probabilities with Laplace smoothing
            self._feature_probabilities[index, :] = (feature_counts + 1) / (feature_counts.sum() + n_features)

    # ------------------- Helper Function to do predicctions for a single datapoint ------------------------
    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        if not issparse(X):
            raise ValueError("Input X must be a sparse matrix")

        # Calculate the log of class prior probabilities
        log_priors = np.log(self._class_priors)

        # Store predictions
        predictions = []

        for row in X:
            # Initialize an array to store log probabilities for each class
            log_probabilities = log_priors.copy()

            for index, cls in enumerate(self._classes):
                # Calculate log-probabilities for each feature given the class
                log_conditional = row.dot(np.log(self._feature_probabilities[index, :].T))
                # Combine with prior log probability
                log_probabilities[index] += log_conditional

            # Predict the class with the highest posterior probability
            predictions.append(self._classes[np.argmax(log_probabilities)])

        return np.array(predictions)