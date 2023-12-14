import numpy as np
import matplotlib.pyplot as plt

from utils import report_performance


# Custom DBSCAN implementation
def custom_dbscan(data, eps, min_samples):
    n, m = data.shape
    labels = np.zeros(n, dtype=int)
    cluster_label = 0
    
    for i in range(n):
        if labels[i] != 0:  # Skip already processed points
            continue
        
        neighbors = np.linalg.norm(data[i] - data, axis=1) < eps
        if np.sum(neighbors) < min_samples:
            labels[i] = -1  # Noise point
        else:
            cluster_label += 1
            labels[i] = cluster_label
            stack = [i]
            
            while stack:
                current_point = stack.pop()
                current_neighbors = np.linalg.norm(data[current_point] - data, axis=1) < eps
                
                for j in range(n):
                    if current_neighbors[j] and labels[j] == 0:
                        labels[j] = cluster_label
                        stack.append(j)
    
    return labels


def main():
    # Generate different data sizes
    data_sizes = [100, 500, 1000, 5000, 10000]

    # Compare performance
    execution_times, memory_usages, silhouette_scores = report_performance(
        data_sizes, custom_dbscan, "DBSCAN - ", {'eps': 3, 'min_samples': 5}
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
    plt.savefig('result_graphs/dbscan_cluster.png')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()