
# ------------------------------------------ Helper Functions ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 3
#
#   This file contains code for implementing helper functions to visulaise principal compoents
#	from PCA and Voronoi regions from KMeans. 
#
#   Description of functions
#
#       - plot_reconstructed_images: 
#			* returns the reconstructed images of MNIST datset with a specific numer of 
#			  principal compoenents 
#			* The image is saved as a .png file. 
#
#       - plot_voronoi: 
#			* Given a clustering algorithms, prints the Voronoi regions with the meshgrid function 
#			* The plot is 2D, but it used Conotur plot to give the color effect 
# --------------------------------------------------------------------------------------------------------------

# ------------------------------------------- Importing Libraries --------------------------------------------

import numpy as np  # type: ignore
import random 
import matplotlib.pyplot as plt  # type: ignore

def plot_reconstructed_images(decomposer, original_images: np.ndarray, image_shape: np.ndarray) -> None:
	eigen_values = decomposer._n_components
	fig, axs = plt.subplots(len(decomposer._n_components), 10, figsize=(15, 8))
	for i, n_components in enumerate(decomposer._n_components):
		decomposer._n_components = n_components
		transformed_images = decomposer.transform(original_images)
		reconstructed_images = decomposer.inverse_transform(transformed_images).reshape(-1, *image_shape)
		for j in range(10):
			axs[i, j].imshow(reconstructed_images[random.randint(0, 999)], cmap='plasma')
			# axs[i, j].axis('off')
		axs[i, 0].set_ylabel(f"{n_components} components", rotation=90, size='large')
		plt.suptitle(f"Reconstruction of Images using {eigen_values} components", fontsize = 24)
		fig.text(.5, .05, f'The images reconstructed with top {max(eigen_values)} principal components can be used for a downstream task', ha='center', fontsize = 20)
	plt.show()
	plt.savefig('Reconstructed_images.png')
	plt.close()

def plot_voronoi_regions(X: np.ndarray, kmeans, k: int) -> None:
    grid_shape = (150, 150) 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_shape[0]),
        np.linspace(y_min, y_max, grid_shape[1])
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    labels_grid = kmeans.predict(grid_points)
    Z = labels_grid.reshape(xx.shape)
    cmap = plt.cm.Paired 
    
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)  
    
    labels = kmeans.predict(X)
    for cluster_id in range(k):
        cluster_points = X[labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            label=f'Cluster {cluster_id+1}', 
            color=cmap((cluster_id / k)), 
            alpha=1
        ) 
    
        plt.scatter(
            kmeans._n_centroids[:, 0], 
            kmeans._n_centroids[:, 1], 
            color='black', 
            marker='x', 
            s=100, 
            label='Cluster Centers'
        )
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Voronoi Regions for K={k}')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f'Voronoi Regions for K={k}.png')
    plt.close()