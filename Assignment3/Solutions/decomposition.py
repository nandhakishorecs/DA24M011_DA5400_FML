# ------------------------------------------ Princiapal Component Analysis -------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 3
#
#   This file contains code for implementing Principal Component Analysis using 
#   Singualar Value Decomposition
#   Description of functions in the class 'PCA':
#       - fit: 
#           * Initialises the class using number of components needed to do PCA 
#           * Decomposes the give  matrix based on its size into Left / Right Singular matrix
#           * The Eigen values of the decomposed matrix are calcualted and sorted in non 
#             decreasing order. The Eigen vectors corresponsing to the top 'k' eigen values are 
#             found. 
#       -   The class has  transform and and inverse transform functions to recreate the data 
#           matrix given a principal component.    
#       -   The class has property functions which plot the signal and noise directions, components 
#           and the top k principal components. 
# -------------------------------------------------------------------------------------------------- 
import numpy as np 
import matplotlib.pyplot as plt 

class PCA:
    __slots__ = '_n_components', '_mean', '_eigenvalues', '_eigenvectors', '_explained_variance', '_cumulative_variance'
    def __init__(self, n_components: int = 15) -> None:
        self._n_components = n_components
        self._mean = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._explained_variance = None
        self._cumulative_variance = None

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape 

        # Normalize the dataset
        self._mean = np.mean(X, axis=0)
        X_normalized = X - self._mean

        # Using inspiration from SVD to reduce time complexity
        if(n_samples > n_features): 
            # Compute the covariance matrix
            covariance_matrix = np.cov(X_normalized, rowvar = False)

            # Compute the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Sort eigenvalues and corresponding eigenvectors in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            self._eigenvalues = eigenvalues[sorted_indices]
            self._eigenvectors = eigenvectors[:, sorted_indices]

            # Compute explained variance
            self._explained_variance = self._eigenvalues / np.sum(self._eigenvalues)
            self._cumulative_variance = np.cumsum(self._explained_variance * 100)

        else: 
            # Calculate X^T.X
            K_matrix = np.dot(X.T, X)

            # Compute the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(K_matrix)

            # Sort eigenvalues and corresponding eigenvectors in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            self._eigenvalues = eigenvalues[sorted_indices]
            self._eigenvectors = eigenvectors[:, sorted_indices]

            # Compute explained variance
            self._explained_variance = self._eigenvalues / np.sum(self._eigenvalues)
            self._cumulative_variance = np.cumsum(self._explained_variance * 100)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_normalized = X - self._mean
        return np.dot(X_normalized, self._eigenvectors[:, :self._n_components])

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        return np.dot(X_transformed, self._eigenvectors[:, :self._n_components].T) + self._mean

    @property
    def explained_varaince_(self) -> None:
        for i, var in enumerate(self._explained_variance[:self._n_components]):
            print(f"Principal Component {i+1}: {var:.4f} (Cumulative Variance: {self._cumulative_variance[i]:.4f})")

    @property
    def plot_components_(self):
        plt.figure(figsize=(15, 8))
        plt.plot(self._eigenvectors[self._n_components:].T)
        plt.title(f'Graph with Signal and Noise components with {self._n_components} components', fontsize = 28)
        plt.ylabel(r'Value of the principal components')
        plt.xlabel(r'Features')
        plt.grid()
        plt.show() 
        plt.savefig(f'Principal_Compoents_Plot_{self._n_components}')
        plt.close()
    
    @property
    def plot_directions_(self): 
        plt.figure(figsize=(15, 8))
        plt.plot(self._eigenvalues[self._n_components:].T, 'r-', label = 'Eigen Values')
        plt.title(f'Graph with Signal and Noise directions with {self._n_components} components', fontsize = 28)
        plt.ylabel(r'Value of the Eigen values')
        plt.xlabel(r'Features')
        plt.legend()
        plt.grid()
        plt.show() 
        plt.savefig(f'Principal_directions_Plot_{self._n_components}')
        plt.close()