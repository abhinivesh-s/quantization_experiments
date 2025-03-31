# Normalize co-occurrence values to the range [0, 1]
min_val = np.min(co_occurrence_matrix)
max_val = np.max(co_occurrence_matrix)

normalized_co_occurrence = (co_occurrence_matrix - min_val) / (max_val - min_val)

# Now convert to distance matrix (1 - normalized value)
distance_matrix = 1 - normalized_co_occurrence




# Step 1: Perform hierarchical clustering using co-occurrence matrix
linkage_matrix = sch.linkage(co_occurrence_matrix, method="ward")

# Step 2: Plot dendrogram to visualize the merging of classes
plt.figure(figsize=(10, 7))
sch.dendrogram(linkage_matrix, labels=class_names, orientation='top')
plt.title("Dendrogram of Class Merges Based on Co-Occurrence")
plt.xlabel("Classes")
plt.ylabel("Distance (Higher = Less Similar)")
plt.show()
