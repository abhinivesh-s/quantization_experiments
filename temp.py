# Step 1: Initialize clusters as individual sets
clusters = {i: {class_names[i]} for i in range(num_classes)}

# ✅ Iterate through each merge step
for merge_id, (i, j, dist, _) in enumerate(linkage_matrix):
    i, j = int(i), int(j)  # Convert indices to integers

    # Merge clusters i and j
    new_cluster = clusters[i] | clusters[j]  # Union of sets
    new_cluster_id = num_classes + merge_id  # Unique new cluster ID

    # Remove old clusters and add new one
    del clusters[i], clusters[j]
    clusters[new_cluster_id] = new_cluster  # Store as set

    # Print merge progress
    print(f"Merging {new_cluster} (Distance: {dist:.4f})")
    print(f"Current Clusters: {list(clusters.values())}\n")






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



def map_to_clusters(y, clusters):
    """Maps each true/pred label to the first element of its cluster (sets version)."""
    cluster_map = {}
    for cluster in clusters.values():  
        rep_class = next(iter(cluster))  # Pick an arbitrary element from the set
        for cls in cluster:
            cluster_map[cls] = rep_class  # Assign representative class

    return np.array([cluster_map[label] for label in y])



# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o', linestyle='-')

# Labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plot of X vs Y")

# Show grid and plot
plt.grid(True)
plt.show()
