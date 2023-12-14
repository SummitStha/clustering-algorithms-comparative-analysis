import numpy as np
import matplotlib.pyplot as plt

from utils import report_performance


# Custom K-Means implementation
def custom_kmeans(data, k, max_iterations=100, tol=1e-4):
    n, m = data.shape
    centroids = data[np.random.choice(n, k, replace=False)]  # Initialize centroids randomly
    prev_centroids = centroids.copy()
    labels = np.zeros(n)
    
    for _ in range(max_iterations):
        # Assign each point to the nearest centroid
        for i in range(n):
            labels[i] = np.argmin(np.linalg.norm(data[i] - centroids, axis=1))
        
        # Update centroids
        for j in range(k):
            centroids[j] = np.mean(data[labels == j], axis=0)
        
        # Check for convergence
        if np.linalg.norm(centroids - prev_centroids) < tol:
            break
        
        prev_centroids = centroids.copy()
    
    return labels


def main():
    # Generate different data sizes
    data_sizes = [100, 500, 1000, 5000, 10000]

    # Compare performance
    execution_times, memory_usages, silhouette_scores = report_performance(
        data_sizes, custom_kmeans, "K-Means - ", {'k': 3}
    )

    # Plotting Results
    plt.figure(figsize=(15, 5))

    # Execution Time Plot
    plt.subplot(1, 3, 1)
    plt.plot(data_sizes, execution_times, marker='o')
    plt.title('Execution Time vs Data Size')
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (s)')
    # plt.legend()
    plt.grid(True)

    # Memory Usage Plot
    plt.subplot(1, 3, 2)
    plt.plot(data_sizes, memory_usages, marker='o', color='orange')
    plt.title('Memory Usage vs Data Size')
    plt.xlabel('Data Size')
    plt.ylabel('Memory Usage (MB)')
    # plt.legend()
    plt.grid(True)

    # Silhouette Score Plot
    plt.subplot(1, 3, 3)
    plt.plot(data_sizes, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score vs Data Size')
    plt.xlabel('Data Size')
    plt.ylabel('Silhouette Score (0-1)')
    # plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig('result_graphs/kmeans_cluster.png')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
