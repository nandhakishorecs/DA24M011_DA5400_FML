# ------------------------------------------ Data Loader ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 3
#
#   This file contains code for Loading MNIST Data and the datapoints from cm_dataset.csv
#   Description of functions:
#       -   load_mnist: 
#               * uses sklearn's datasets function to fetch the dataset from the given url in 
#                 the question paper. 
#               * samples 100 images randomly from each class and gives out a 1000 sample dataset.
#               * The images are normalised for computational ease. 
#
#       -   load_csv_file: 
#               * reads the csv and returns a nump array 
#               * Column names are added for handling data with ease. 
# --------------------------------------------------------------------------------------------------------

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml

def load_mnist(n_samples:int = 100):
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist.data, mnist.target.astype(int)

    # Sample 100 images per class
    x_sample = []
    y_sample = []

    for i in range(10):  # For each digit class (0-9)
        indices = np.where(y == i)[0]
        sampled_indices = np.random.choice(indices, n_samples, replace=False)
        x_sample.append(x[sampled_indices])
        y_sample.append(y[sampled_indices])

    # Combine sampled data
    x_sample = np.vstack(x_sample)
    y_sample = np.hstack(y_sample)

    # Shuffle the dataset
    x_sample, y_sample = shuffle(x_sample, y_sample, random_state=42)

    # Normalize pixel values
    x_sample = x_sample / 255.0

    return x_sample

def load_csv_file(file_path: str) -> np.ndarray: 
    data = pd.read_csv(file_path, header = None)
    data.columns = ['0', '1']
    data = data.to_numpy() 
    return data