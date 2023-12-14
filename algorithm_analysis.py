import matplotlib.pyplot as plt

from kmeans import custom_kmeans
from dbscan import custom_dbscan
from utils import clean_data, perform_clustering, visualize_clusters


# Function to compare execution times and memory usage
def compare_performance(data_sizes, clustering_function1, clustering_function2, kwargs1, kwargs2):
    execution_times1 = []
    execution_times2 = []
    memory_usage1 = []
    memory_usage2 = []
    silhouette_score1 = []
    silhouette_score2 = []

    scaled_data = clean_data('dataset/credit_card_data.csv')

    for size in data_sizes:
        sample_data = scaled_data[:size]

        # Clustering with function 1
        kmeans_labels, time1, kmeans_silhouette, memory1 = perform_clustering(sample_data, clustering_function1, **kwargs1)
        execution_times1.append(time1)
        memory_usage1.append(memory1)
        silhouette_score1.append(kmeans_silhouette)

        # Clustering with function 2
        dbscan_labels, time2, dbscan_silhouette, memory2 = perform_clustering(sample_data, clustering_function2, **kwargs2)
        execution_times2.append(time2)
        memory_usage2.append(memory2)
        silhouette_score2.append(dbscan_silhouette)

        # Plot Custom K-Means results
        visualize_clusters(sample_data, kmeans_labels, title=f"K-Means - {size}")

        # Plot Custom DBSCAN results
        visualize_clusters(sample_data, dbscan_labels, title=f"DBSCAN - {size}")

    return execution_times1, execution_times2, memory_usage1, memory_usage2, silhouette_score1, silhouette_score2


def main():
    # Generate different data sizes
    data_sizes = [100, 500, 1000, 5000, 10000]

    # Compare performance
    kmeans_times, dbscan_times, kmeans_memory, dbscan_memory, kmeans_silhouette, dbscan_silhouette = compare_performance(
        data_sizes, custom_kmeans, custom_dbscan, {'k': 3}, {'eps': 3, 'min_samples': 5}
    )

    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot Execution Times
    axes[0].plot(data_sizes, kmeans_times, label='K-Means', marker='o')
    axes[0].plot(data_sizes, dbscan_times, label='DBSCAN', marker='o')
    axes[0].set_title('Execution Times Comparison')
    axes[0].set_xlabel('Data Size')
    axes[0].set_ylabel('Execution Time (seconds)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Memory Usage
    axes[1].plot(data_sizes, kmeans_memory, label='K-Means', marker='o')
    axes[1].plot(data_sizes, dbscan_memory, label='DBSCAN', marker='o')
    axes[1].set_title('Memory Usage Comparison')
    axes[1].set_xlabel('Data Size')
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Silhouette Score
    axes[2].plot(data_sizes, kmeans_silhouette, label='K-Means', marker='o')
    axes[2].plot(data_sizes, dbscan_silhouette, label='DBSCAN', marker='o')
    axes[2].set_title('Silhouette Score Comparison')
    axes[2].set_xlabel('Data Size')
    axes[2].set_ylabel('Silhouette Score (0-1)')
    axes[2].legend()
    axes[2].grid(True)

    ###### Single Plots #######
    # Plotting Results
    # plt.figure(figsize=(8, 4))

    # Execution Time Plot
    # plt.subplot(1, 3, 1)
    # plt.plot(data_sizes, kmeans_times, label='K-Means', marker='o')
    # plt.plot(data_sizes, dbscan_times, label='DBSCAN', marker='o')
    # plt.title('Execution Time vs Data Size')
    # plt.xlabel('Data Size')
    # plt.ylabel('Execution Time (s)')
    # plt.legend()
    # plt.grid(True)

    # # Memory Usage Plot
    # plt.subplot(1, 3, 2)
    # plt.plot(data_sizes, kmeans_memory, label='K-Means', marker='o')
    # plt.plot(data_sizes, dbscan_memory, label='DBSCAN', marker='o')
    # plt.title('Memory Usage vs Data Size')
    # plt.xlabel('Data Size')
    # plt.ylabel('Memory Usage (MB)')
    # plt.legend()
    # plt.grid(True)

    # # Silhouette Score Plot
    # plt.subplot(1, 3, 3)
    # plt.plot(data_sizes, kmeans_silhouette, label='K-Means', marker='o')
    # plt.plot(data_sizes, dbscan_silhouette, label='DBSCAN', marker='o')
    # plt.title('Silhouette Score vs Data Size')
    # plt.xlabel('Data Size')
    # plt.ylabel('Silhouette Score (0-1)')
    # plt.legend()
    # plt.grid(True)

    # Adjust space between vertical subplots
    plt.subplots_adjust(wspace=2)

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig('result_graphs/clustering_performance_comparison.png')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
