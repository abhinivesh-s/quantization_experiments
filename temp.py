import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import copy # To keep track of clusters without modifying originals

# --- 1. Prepare Data ---
# Example Co-occurrence Matrix (Same as before)
class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F']
co_occurrence_matrix = np.array([
    # A   B   C    D    E    F
    [100, 80, 70,  10,   5,  15], # A
    [ 80, 90, 65,  12,   8,  10], # B
    [ 70, 65, 85,   8,  10,   5], # C
    [ 10, 12,  8, 110,  85,  75], # D
    [  5,  8, 10,  85, 100,  80], # E
    [ 15, 10,  5,  75,  80,  95]  # F
])

n_classes = co_occurrence_matrix.shape[0]

# --- 2. Convert Similarity (Co-occurrence) to Dissimilarity (Distance) ---
# distance = max(similarity) - similarity
# Ensure diagonal is 0, matrix is symmetric and non-negative
max_similarity = np.max(co_occurrence_matrix[np.triu_indices(n_classes, k=1)])
distance_matrix = max_similarity - co_occurrence_matrix
distance_matrix[distance_matrix < 0] = 0
distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix, 0)

print("--- Distance Matrix (Lower value means more similar/higher co-occurrence) ---")
print(np.round(distance_matrix, 2))

# --- 3. Calculate Linkage ---
# Convert the square distance matrix to a condensed distance matrix (1D array)
# SciPy's linkage function requires this format.
condensed_distance = squareform(distance_matrix)

# Perform hierarchical/agglomerative clustering
# Linkage methods determine how distance between clusters is calculated:
# - 'average': Uses the average of the distances of each observation of the two sets. Good for co-occurrence.
# - 'complete': Uses the maximum distances between all observations of the two sets.
# - 'single': Uses the minimum of the distances between all observations of the two sets.
# - 'ward': Minimizes the variance of the clusters being merged. (Requires Euclidean-like distances)
# Choose 'average' as it reflects the average co-occurrence idea well.
linkage_method = 'average'
Z = linkage(condensed_distance, method=linkage_method)

print(f"\n--- Linkage Matrix (Z) using '{linkage_method}' linkage ---")
print("Format: [idx1, idx2, distance, num_items_in_new_cluster]")
print(Z)

# --- 4. Interpret Linkage Matrix - Show Clusters Iteratively ---

print("\n--- Iterative Clustering Steps ---")

# Start with each class as its own cluster
# We'll store clusters as sets of *original* class indices (0 to n_classes-1)
initial_clusters = [{i} for i in range(n_classes)]
current_clusters = copy.deepcopy(initial_clusters) # List of active clusters (sets)

# Map cluster indices (from linkage matrix) to the set of original items
# Indices 0 to n-1 are original items
# Indices n to 2n-2 are newly formed clusters from linkage matrix rows
cluster_map = {i: {i} for i in range(n_classes)}

# Iterate through the linkage matrix rows (each row represents a merge)
for i in range(Z.shape[0]):
    row = Z[i]
    idx1, idx2, dist, num_items = int(row[0]), int(row[1]), row[2], int(row[3])

    # Find the actual sets of original items for the merging clusters
    set1 = cluster_map[idx1]
    set2 = cluster_map[idx2]

    # Create the new merged cluster
    new_set = set1.union(set2)

    # Assign this new set to the new cluster index (n + i)
    new_cluster_idx = n_classes + i
    cluster_map[new_cluster_idx] = new_set

    # Update the list of *current* clusters
    # Find and remove the two old clusters from the list
    temp_current_clusters = []
    merged_one = False
    merged_two = False
    for cluster in current_clusters:
        if cluster == set1 and not merged_one:
             merged_one = True # Remove first occurrence
        elif cluster == set2 and not merged_two:
             merged_two = True # Remove first occurrence
        else:
            temp_current_clusters.append(cluster)

    # Add the new merged cluster
    temp_current_clusters.append(new_set)
    current_clusters = temp_current_clusters

    # --- Output the state at this step ---
    print(f"\nStep {i+1}:")
    # Convert indices to names for printing
    names1 = sorted([class_names[item_idx] for item_idx in set1])
    names2 = sorted([class_names[item_idx] for item_idx in set2])
    print(f" - Merged: Cluster {idx1} {names1} and Cluster {idx2} {names2}")
    print(f" - Distance: {dist:.2f}")
    print(f" - New Cluster ID: {new_cluster_idx} (Size: {num_items})")
    print(f" - Current Clusters ({len(current_clusters)}):")
    # Print current clusters with names
    current_clusters_named = []
    for cluster_set in current_clusters:
         current_clusters_named.append(sorted([class_names[item_idx] for item_idx in cluster_set]))
    # Sort the list of lists for consistent display order (optional)
    current_clusters_named.sort(key=lambda x: x[0])
    print("   ", current_clusters_named)


# --- 5. Visualize ---
# The Dendrogram perfectly visualizes this iterative merging process
plt.figure(figsize=(12, 7))
dendrogram(
    Z,
    labels=class_names,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=10.,  # font size for the x axis labels
    orientation='top', # Can be 'top', 'bottom', 'left', 'right'
)
plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
plt.xlabel('Class')
plt.ylabel('Distance (Derived from Co-occurrence)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n--- End of Iterative Clustering ---")
