# Step 8: Get cluster labels for a given threshold
num_clusters = 10  # You can adjust this to control granularity
clusters = sch.fcluster(linked, num_clusters, criterion="maxclust")

# Step 9: Count cluster sizes
cluster_counts = Counter(clusters)

# Step 10: Filter clusters that contain 4 or fewer elements
filtered_clusters = {c: [] for c in cluster_counts if cluster_counts[c] <= 4}

# Step 11: Collect class names belonging to filtered clusters
for class_name, cluster_id in zip(class_names, clusters):
    if cluster_id in filtered_clusters:
        filtered_clusters[cluster_id].append(class_name)

# Step 12: Print filtered clusters
print("Clusters with 4 or Fewer Classes:")
for cluster_id, class_group in filtered_clusters.items():
    print(f"Cluster {cluster_id}: {class_group}")
