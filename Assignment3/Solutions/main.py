# ------------------------------------------ PCA & KMeans ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundation of Machine Learning Assignment 3

# ----------------------------------------- Importing Libraries -----------------------------------------------
# Custom written function 
from helper_functions import *
from decomposition import * 
from clustering import *
from data_loader import *

# Ignore warnings 
import warnings 
warnings.filterwarnings("ignore")

# ---------------------------------------- Main Function ------------------------------------------------------
if __name__ == "__main__":
      
    # ------------------------------------------- Personal Data ----------------------------------------
    print("\nDA5400 - Foundation of Machine Learning\n")
    print("\nAssignment 3\n")
    print("\nSubmitted by Nandhakishore C S - DA24M011\n")

    
    # --------------------------- PCA - Principal Component Analysis ------------------------------------
    print('\nPRINCICPAL COMPONENT ANALYSIS\n')    

    # Load MNIST Dataset 
    x_sample = load_mnist(n_samples = 100)

    
    # 1. (i). Run PCA 
    pca = PCA(n_components = 16)
    pca.fit(x_sample)
    transformed_images = pca.transform(x_sample)

    # Plotting all principal directions 
    pca.plot_directions_
    # Plotting all principal components
    pca.plot_components_
    # Print explained variance
    pca.explained_varaince_

    # Prinitng Top K Principal Components 
    fig = plt.figure(figsize=(30, 20))
    for index in range(0, pca._n_components):
        plt.subplot(int(pca._n_components/4), 4, index + 1)
        plt.plot(pca._eigenvectors.T[index])
        plt.title(f'Eigen Vector for $ \lambda $ = {pca._eigenvalues[index]:.2f}')
        plt.grid() 
        plt.xlabel('Features')
        plt.ylabel('Principal Component Values')
        # plt.show() 
    plt.suptitle('Principal components for top 16 components ', fontsize = 40)
    plt.savefig('Eigen_vectors_plot.png', bbox_inches = 'tight')    
    plt.close() 

    print('\nThe images are saved in the same directory\n')

    # 1. (ii). Reconstruction of MNIST using PCA with n_components = 75
    # Reconstruct and visualize images using different numbers of components
    pca._n_components = [10, 25, 50, 75]
    plot_reconstructed_images(
        decomposer = pca, 
        original_images = x_sample, 
        image_shape = (28, 28)
    )

    # ---------------------------------- LLyod's Algorithm - KMeans Clustering ------------------------------------
    print('\nK MEANS CLUSTERING\n')    
    print('\nThe images are saved in the same directory\n')
    # Loading Dataset 
    X = load_csv_file(
        file_path = '/cm_dataset_2.csv'
    )
    
    # 2. (i). Doing Lloyd's algorithm for 5 different initialisations 
    errors = [] 
    for i in range(5):
        kmeans = clustering(k = 2, random_state = i + 123)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        
        # getting errors 
        errors.append(kmeans._errors)

        for cluster_id in range(kmeans._K): 
            cluster_points = X[labels == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1], 
                label=f'Cluster {cluster_id+1}'
            )
        # plotting the clusters 
        plt.scatter(
            kmeans._n_centroids[:, 0], 
            kmeans._n_centroids[:, 1], 
            color = 'black', 
            marker = 'x', 
            s = 100, 
            label = 'Centroids'
        )
        plt.title(f'Initialization {i+1}: Clusters')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()
        plt.savefig(f'Initialization {i+1}: Clusters.png')
        plt.close()

    # Plotting the errors 
    for i in range(5): 
        plt.plot(errors[i], label = f'error for initialisation {i+1}')

    plt.title('Error Function for five different initialisations vs # Iterations')
    plt.grid() 
    plt.show()
    plt.legend()
    plt.savefig('errors.png')
    plt.close()

    # 2 . (ii) Printing Vornoi regions for k = 2,3,4,5 with arbitrary initialisation 

    K_clusters = [2, 3, 4, 5]
    for k in K_clusters:
        kmeans = clustering(k = k) 
        kmeans.fit(X)
        plot_voronoi_regions(X, kmeans, k)
    
    # 3. (iii) Since the data set is not linearly separable and is in a manifold, we use Kernel Kmeans
    # This is done by mapping data to higher dimension using RBF and performing K Means on it. 

    from sklearn.cluster import SpectralClustering # type: ignore

    # Step 1: Perform Spectral Clustering using sklearn
    spectral_clustering = SpectralClustering(
        n_clusters=2,
        affinity='rbf',
        gamma=1,
        random_state=42
    )
    labels_spectral = spectral_clustering.fit_predict(X)

    # Step 2: Plot the clusters
    plt.figure(figsize=(8, 6))
    for cluster in range(2):
        cluster_points = X[labels_spectral == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster + 1}")

    plt.title("Spectral Clustering (using sklearn) in cm_dataset.csv")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('Spectral_Clusters.png')
    plt.close() 
