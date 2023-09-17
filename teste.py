import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

fileimport = "database_gas.csv"

data=pd.read_csv(fileimport)
data['temperature'] = pd.to_numeric(data['temperature'], errors='coerce')
data['humidity'] = pd.to_numeric(data['humidity'], errors='coerce')
data.dropna(inplace=True)

data_values = data[['temperature', 'humidity']].values

num_clusters = 5
fuzziness = 3

U = np.random.rand(data_values.shape[0], num_clusters)
U /= np.sum(U, axis=1)[:, np.newaxis]

def calculate_centroids(data, U, fuzziness):
    membership_powers = U ** fuzziness
    centroids = np.dot(membership_powers.T, data) / np.sum(membership_powers, axis=0)[:, np.newaxis]
    return centroids  

def calculate_membership(data, centroids, fuzziness):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    membership_denominator = distances ** (2 / (fuzziness - 1))
    U_new = 1 / np.sum((1 / membership_denominator)[:, :, np.newaxis], axis=1)
    return U_new

centroids = np.random.rand(num_clusters, data_values.shape[1])

max_iterations = 100
threshold = 0.001

for iteration in range(max_iterations):
    # Calculate new membership values
    U_new = calculate_membership(data_values, centroids, fuzziness)

    # Calculate new centroids
    centroids = calculate_centroids(data_values, U_new, fuzziness)

    # Check for convergence
    if np.linalg.norm(U_new - U) < threshold:
        print(f"Converged after {iteration} iterations.")
        break

    # Update membership matrix for the next iteration
    U = U_new

def kmeans(X, K):
    # 1. Randomly initialize the centroids
    centroids = X[np.random.choice(range(X.shape[0]), size=K, replace=False), :]
    
    while True:
        # 2. Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # 3. Recalculate the centroids as the mean of the current clusters
        new_centroids = np.array([X[labels==k].mean(axis=0) for k in range(K)])
        
        # If the centroids aren't moving anymore it is time to stop
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Elbow method
distortions = []
K_range = range(1,11)
for k in K_range:
    _, centroids = kmeans(data_values, k)
    distortions.append(sum(np.min(np.linalg.norm(data_values - centroids[_], axis=1)) for _ in range(k)) / data_values.shape[0])

# Plot the elbow
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


cluster_labels = np.argmax(U_new, axis=1)

print("Cluster Centroids:")
print(centroids)

plt.scatter(data.temperature, data.humidity, c=cluster_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Fuzzy C-Means Clustering Results')
plt.legend()
plt.show()

