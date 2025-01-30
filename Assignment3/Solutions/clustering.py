# ------------------------------------------ K Means Clustering ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 3
#
#   This file contains code for implementing Clustering using Lloyd's algorithm 
#   Description of functions in the class 'clustering':
#
#       - The inputs for the classes are as follows: 
#           *   X: dense matrix with text (np.ndarray)
#           *   When called, the class is initialised with k = 2 and a default random seed for initialising 
#               cluster centroids. 
#
#       - fit (X) 
#           *   Finds the means of clusters and assigns data to 'k' clusters 
#           *   the clusters are assigned with labels from {0, 1, 2, ... , k-1}
#   
#       - predict(X):
#           *   Returns the label (cluster number) for plotting and downstream taks 

# --------------------------------------------------------------------------------------------------------------
# Imporing Libraries 
import numpy as np 

class clustering:
    # --------------------------------------- Class Initialisation ------------------------------------
    __slots__ = '_K', '_max_iterations', '_n_centroids', '_errors'
    def __init__(self, k:int = 2, max_iter:int = 100, random_state = 42) -> None:
        self._K = k
        self._max_iterations = max_iter
        self._n_centroids = k * [None]
        self._errors=[]
        np.random.seed(random_state)

    # ----------------------------- Fit Method to assign centroids to clusters  ------------------------------------
    def fit(self, X:np.ndarray) -> None:
        random_index = np.random.choice(len(X), self._K, replace=False)
        self._n_centroids = X[random_index]

        for _ in range(self._max_iterations):
            distances = np.linalg.norm(X[:,np.newaxis] - self._n_centroids , axis = 2)
            labels = np.argmin(distances,axis=1)

            # update centroids by calculating the mean of cluster
            new_centroids = []
            for i in range(self._K):
                new_centroids.append(X[labels == i].mean(axis=0))

            new_centroids = np.array(new_centroids)

            # compute error using llyod's algorithms objective (distance between data and cluster mean)
            current_error = np.sum((X - self._n_centroids[labels]) ** 2)
            self._errors.append(current_error)

            # check for assignment - stop when means are not changing 
            if np.allclose(self._n_centroids, new_centroids):
                break

            self._n_centroids = new_centroids

    # ------------------------ Predict method to return cluster numbers ------------------------------------
    def predict(self, X: np.ndarray) ->np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - self._n_centroids, axis = 2)
        labels = np.argmin(distances, axis = 1)
        return labels