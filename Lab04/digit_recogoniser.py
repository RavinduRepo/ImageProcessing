import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
train_data = pd.read_csv('/e:/My Projects/python/Image Processing/Lab04/digit_recognizer/train.csv')
test_data = pd.read_csv('/e:/My Projects/python/Image Processing/Lab04/digit_recognizer/test.csv')

# Normalize and reshape data
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Show some data examples
def show_examples(data, labels, num_examples=5):
    for i in range(num_examples):
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.show()

show_examples(X_train, y_train)

# Elbow method to identify optimal K value
def elbow_method(data):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    optimal_k = distortions.index(min(distortions)) + 1
    return optimal_k

optimal_k = elbow_method(X_train_scaled)

# Clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k)
clusters = kmeans.fit_predict(X_train_scaled)

# Visualize initial clusters
def visualize_clusters(data, clusters, num_examples=5):
    for i in range(num_examples):
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title(f'Cluster: {clusters[i]}')
        plt.show()

visualize_clusters(X_train, clusters)

# Show random samples with their respective cluster
def show_random_samples(data, clusters, num_samples=5):
    indices = np.random.choice(len(data), num_samples, replace=False)
    for i in indices:
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title(f'Cluster: {clusters[i]}')
        plt.show()

show_random_samples(X_train, clusters)

# Silhouette method to identify optimal K value
def silhouette_method(data):
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(data)
        score = silhouette_score(data, clusters)
        silhouette_scores.append(score)
    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('The Silhouette Method showing the optimal k')
    plt.show()
    optimal_k_silhouette = K[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k_silhouette

optimal_k_silhouette = silhouette_method(X_train_scaled)

# Clustering with optimal K from silhouette method
kmeans_silhouette = KMeans(n_clusters=optimal_k_silhouette)
clusters_silhouette = kmeans_silhouette.fit_predict(X_train_scaled)

# Visualize clusters after silhouette method
visualize_clusters(X_train, clusters_silhouette)

# Show random samples with their respective cluster after silhouette method
show_random_samples(X_train, clusters_silhouette)

# PCA for 2D visualization
def pca_2d_visualization(data, clusters, title='PCA 2D Visualization'):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    plt.figure(figsize=(8, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Visualize initial clusters in 2D
pca_2d_visualization(X_train_scaled, clusters, title='PCA 2D Visualization of Initial Clusters')

# Visualize clusters after silhouette method in 2D
pca_2d_visualization(X_train_scaled, clusters_silhouette, title='PCA 2D Visualization of Clusters after Silhouette Method')
