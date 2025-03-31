
# --- 3. Helper Function to Find Original Cluster Members ---

# cluster_idx: index from linkage matrix (can be < N for original items, >= N for merged clusters)
# Z: linkage matrix
# N: number of original items
# labels: list of original labels
# cluster_map: dictionary acting as cache/memoization for performance
def get_cluster_elements(cluster_idx, Z, N, labels, cluster_map):
    """Recursively finds the original labels belonging to a cluster index."""
    cluster_idx = int(cluster_idx) # Ensure it's an integer index

    # Check cache
    if cluster_idx in cluster_map:
        return cluster_map[cluster_idx]

    # Base case: If it's an original item (index < N)
    if cluster_idx < N:
        result = {labels[cluster_idx]} # Return a set with the single label
        cluster_map[cluster_idx] = result
        return result
    # Recursive case: If it's a merged cluster (index >= N)
    else:
        # Find the row in Z that formed this cluster
        # The cluster with index (N + i) was formed at row i of Z
        merge_row_idx = cluster_idx - N
        if merge_row_idx >= len(Z):
             raise ValueError(f"Invalid cluster index {cluster_idx} derived from Z.")

        # Get the indices of the clusters that were merged
        idx1 = Z[merge_row_idx, 0]
        idx2 = Z[merge_row_idx, 1]

        # Recursively find elements of the two merged sub-clusters
        elements1 = get_cluster_elements(idx1, Z, N, labels, cluster_map)
        elements2 = get_cluster_elements(idx2, Z, N, labels, cluster_map)

        # The new cluster contains the union of elements
        result = elements1.union(elements2)
        cluster_map[cluster_idx] = result
        return result

# --- 4. Iteratively Find Clusters and Members ---

print("Iterative Cluster Formation (Sorted by Ward distance - lowest variance increase first):")
print("-" * 70)
print(f"{'Step':<5} | {'Ward Distance':<15} | {'Cluster Size':<12} | {'Cluster Members'}")
print(f"{'(inc. variance)':<5} | {'':<15} | {'':<12} | {'(Original Labels)'}")
print("-" * 70)

# Cache for cluster elements to avoid redundant calculations
cluster_elements_cache = {}

# Iterate through the linkage matrix rows (already sorted by distance/variance increase)
for i in range(len(Z)):
    row = Z[i]
    # The 'distance' here is the Ward variance increase
    ward_distance = row[2]
    num_items_in_cluster = int(row[3])

    # The index of the newly formed cluster is N + i
    new_cluster_idx = N + i

    # Find the original members of this newly formed cluster
    current_cluster_members = get_cluster_elements(
        new_cluster_idx, Z, N, labels, cluster_elements_cache
    )

    # Sort members for consistent printing
    members_str = ", ".join(sorted(list(current_cluster_members)))

    print(f"{i+1:<5} | {ward_distance:<15.4f} | {num_items_in_cluster:<12} | {{{members_str}}}")

print("-" * 70)
print("Note: 'Ward Distance' represents the increase in within-cluster variance when merging.")
print("Lower values indicate merges preferred by the Ward criterion.")

# --- Optional: Visualize using a Dendrogram ---
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram (Ward method on Co-occurrence Counts)')
    plt.xlabel('Class Index / Cluster')
    plt.ylabel('Ward Distance (Increase in Variance)')
    sch.dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90.,
        leaf_font_size=10.,
    )
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\nInstall matplotlib to visualize the dendrogram: pip install matplotlib")
except Exception as e:
    print(f"\nCould not plot dendrogram: {e}")
