import numpy as np
import scipy.cluster.hierarchy as sch

# Example: Simulated co-occurrence matrix (replace with actual data)
num_classes = 10  # Example: 10 classes
np.random.seed(42)
co_occurrence_matrix = np.random.rand(num_classes, num_classes)
co_occurrence_matrix = (co_occurrence_matrix + co_occurrence_matrix.T) / 2  # Make it symmetric

# Compute linkage
linkage_matrix = sch.linkage(co_occurrence_matrix, method="ward")

# Step 1: Initialize each class as its own cluster
clusters = {i: {i} for i in range(num_classes)}  # {cluster_id: set of class indices}

# Step 2: Iteratively merge clusters based on linkage matrix
for merge_id, (i, j, dist, _) in enumerate(linkage_matrix):
    i, j = int(i), int(j)  # Convert to integer indices

    # Merge clusters i and j
    new_cluster = clusters[i] | clusters[j]  # Union of both sets
    new_cluster_id = num_classes + merge_id  # Unique new cluster ID

    # Remove old clusters and add new one
    del clusters[i], clusters[j]
    clusters[new_cluster_id] = new_cluster

    # Print clusters at each step
    print(f"Step {merge_id + 1}: Merged {i} and {j} → New Cluster {new_cluster_id}")
    print(f"Clusters: {list(clusters.values())}\n")
