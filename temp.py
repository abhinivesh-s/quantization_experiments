# Step 1: Perform hierarchical clustering using co-occurrence matrix
linkage_matrix = sch.linkage(co_occurrence_matrix, method="ward")

# Step 2: Plot dendrogram to visualize the merging of classes
plt.figure(figsize=(10, 7))
sch.dendrogram(linkage_matrix, labels=class_names, orientation='top')
plt.title("Dendrogram of Class Merges Based on Co-Occurrence")
plt.xlabel("Classes")
plt.ylabel("Distance (Higher = Less Similar)")
plt.show()
