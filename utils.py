import time
import psutil

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def clean_data(data_path):
    # Load the Credit Card Dataset for Clustering
    # Download the dataset from Kaggle: https://www.kaggle.com/arjunbhasin2013/ccdata
    data = pd.read_csv(data_path)

    # Drop irrelevant columns
    data.drop(['CUST_ID'], axis=1, inplace=True)
    data.fillna(method='ffill', inplace=True)  # Fill missing values with the previous values

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


# Function to measure memory usage
def memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert to megabytes


# Function to perform clustering and measure performance
def perform_clustering(data, clustering_function, **kwargs):
    start_time = time.time()
    # start_memory = memory_usage()
    labels = clustering_function(data, **kwargs)
    # end_memory = memory_usage()
    end_time = time.time()
    # memory = end_memory - start_memory
    execution_time = end_time - start_time
    silhouette = silhouette_score(data, labels)
    memory = memory_usage()

    print(f"Report for {clustering_function.__name__} - {len(data)}")
    print(f"Execution time: {execution_time}")
    print(f"Memory Usage: {memory}")
    print(f"Silhouette Score: {silhouette}")
    print("---"*40)
    
    return labels, execution_time, silhouette, memory


# Function to visualize the clusters
def visualize_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.show()


# Function to compare execution times and memory usage
def report_performance(data_sizes, clustering_function, title, kwargs):
    execution_times = []
    memory_usages = []
    silhouette_scores = []

    scaled_data = clean_data('dataset/credit_card_data.csv')

    for size in data_sizes:
        size_title = title + str(size)
        sample_data = scaled_data[:size]

        # Clustering with function 1
        labels, execution_time, silhouette_score, memory_usage = perform_clustering(sample_data, clustering_function, **kwargs)
        execution_times.append(execution_time)
        memory_usages.append(memory_usage)
        silhouette_scores.append(silhouette_score)

        # Plot Custom K-Means results
        visualize_clusters(sample_data, labels, size_title)

    return execution_times, memory_usages, silhouette_scores
