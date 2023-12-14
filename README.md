## Comparative Analysis of K-Means and DBSCAN Clustering Algorithms

#### Dataset Used
- “Credit Card Dataset for Clustering” from Kaggle
- https://www.kaggle.com/datasets/arjunbhasin2013/ccdata/data
- Can be found under the <em>dataset</em> folder

#### Files and Folders Details
- <i> dataset </i> folder contains the dataset used for this comparative study.
- <i> result_graphs </i> folder contains the graphs generated from the analysis scripts
- <i> utils.py </i> file contains all the common utilities functions
- <i> kmeans.py </i> file contains the implementation of the K-Means Clustering Algorithm with its separate analysis. This script can be run independently to get the clusters and evaluate the performance (execution times, memory usage, silhouette score)
- <i> dbscan.py </i> file contains the implementation of the DBSCAN Clustering Algorithm with its separate analysis. This script can be run independently to get the clusters and evaluate the performance (execution times, memory usage, silhouette score)
- <i> algorithm_analysis.py </i> file contains the main script that reads the dataset as input, partitions it into different sizes, and then runs both the K-Means and DBSCAN algorithms on these different data sizes. It the compares the performance of the two algorithms and plots the graphs for comparison.


#### Metrics Reported:
1. Execution Time:
- Description: Measures the time taken for each algorithm to complete the clustering process.
- Analysis Use Case: Lower execution time indicates faster performance, making it a crucial metric for assessing algorithm efficiency.
2. Memory Usage:
- Description: Tracks the memory consumption during the execution of each algorithm.
- Analysis Use Case: Lower memory usage is favorable as it indicates efficient resource utilization, especially important for large datasets.
3. Silhouette Score (Accuracy):
- Description: Evaluates the quality of clusters by measuring the cohesion within clusters and separation between clusters.
- Analysis Use Case: Higher silhouette scores indicate well-defined, distinct clusters, providing a quantitative measure of clustering effectiveness.
4. Data Size Variations (Scalability Analysis):
- Description: Tests the algorithms on datasets of different sizes.
- Analysis Use Case: Assesses scalability and generalizability of the algorithms across varying data sizes and compares all of the above three metrics.


#### Execution Instructions:
1. Create a virtual environment
```
python -m venv <virtual_env_name>
```

2. Activate the virtual environment
```
source <virtual_env_name>/bin/activate (Linux and Mac)

.\<virtual_env_name>\Scripts\activate (Windows)
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Execute the program
```
python algorithm_analysis.py
```
