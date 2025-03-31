import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AffinityPropagation, SpectralClustering, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# --- 1. Prepare Data ---
# Example Co-occurrence Matrix (Replace with your actual data)
# Let's assume 6 classes: A, B, C, D, E, F
# Clusters might be {A, B, C} and {D, E, F}

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

# --- Optional: Ensure Symmetry (if your matrix isn't perfectly symmetric) ---
# co_occurrence_matrix = (co_occurrence_matrix + co_occurrence_matrix.T) / 2

# --- Convert Similarity (Co-occurrence) to Dissimilarity (Distance) ---
# Required for some algorithms (Agglomerative, MDS)
# Common approach: distance = max(similarity) - similarity
# Handle diagonal: Often set self-distance to 0
max_similarity = np.max(co_occurrence_matrix[np.triu_indices(n_classes, k=1)]) # Max off-diagonal
# or use np.max(co_occurrence_matrix) if diagonal represents meaningful self-similarity
distance_matrix = max_similarity - co_occurrence_matrix
np.fill_diagonal(distance_matrix, 0) # Set self-distance to zero

# Ensure distance matrix is valid (non-negative, symmetric, zero diagonal)
distance_matrix[distance_matrix < 0] = 0 # Clip any potential negative values
distance_matrix = (distance_matrix + distance_matrix.T) / 2 # Ensure symmetry
np.fill_diagonal(distance_matrix, 0)


print("--- Original Co-occurrence Matrix ---")
print(co_occurrence_matrix)
print("\n--- Derived Distance Matrix ---")
print(np.round(distance_matrix, 2))

# --- 2/3. Perform Clustering ---

print("\n--- Clustering Results ---")

# Method 1: Affinity Propagation (works directly on similarity/affinity)
# Preference controls how many exemplars (clusters) are found. -np.inf means auto.
# Damping factor helps convergence.
# Note: Affinity propagation prefers *negative* distances or *positive* similarities.
# Let's use the original co-occurrence matrix as similarity.
# We might need to adjust the diagonal for AP, as it influences the 'preference'
# Let's try with the original co-occurrence first.
try:
    # Using affinity='precomputed' requires a similarity matrix
    # If AP expects dissimilarity, use -distance_matrix.
    # Let's assume it works well with the positive co-occurrence as similarity:
    af_prop = AffinityPropagation(affinity='precomputed', damping=0.7, random_state=42)
    af_prop.fit(co_occurrence_matrix) # Provide similarity matrix
    labels_ap = af_prop.labels_
    n_clusters_ap = len(np.unique(labels_ap))
    print(f"Affinity Propagation found {n_clusters_ap} clusters.")
    print("Labels:", labels_ap)
except Exception as e:
    print(f"Affinity Propagation failed: {e}. Might need parameter tuning or data scaling.")
    labels_ap = np.zeros(n_classes, dtype=int) # Placeholder


# Method 2: Spectral Clustering (works directly on affinity/similarity)
# Requires specifying n_clusters
n_clusters_spectral = 2 # Guess or determine based on data/problem
# Can use affinity='precomputed' with the co-occurrence matrix
sc = SpectralClustering(n_clusters=n_clusters_spectral,
                        affinity='precomputed', # Use the similarity matrix
                        assign_labels='kmeans', # Or 'discretize'
                        random_state=42)
labels_sc = sc.fit_predict(co_occurrence_matrix) # Provide similarity matrix
print(f"\nSpectral Clustering ({n_clusters_spectral} clusters):")
print("Labels:", labels_sc)


# Method 3: Agglomerative Clustering (Hierarchical - needs distance)
# Requires specifying n_clusters or distance_threshold
n_clusters_agg = 2 # Guess or determine based on data/problem
# Needs distance matrix if using metric='precomputed'
# Linkage methods: 'ward', 'complete', 'average', 'single'
agg = AgglomerativeClustering(n_clusters=n_clusters_agg,
                              affinity='precomputed', # Use the distance matrix
                              linkage='average') # 'average' often works well for this type
labels_agg = agg.fit_predict(distance_matrix) # Provide distance matrix
print(f"\nAgglomerative Clustering ({n_clusters_agg} clusters, average linkage):")
print("Labels:", labels_agg)

# --- 4. Visualize Results ---

# Choose one set of labels for consistent visualization (e.g., from Agglomerative)
cluster_labels = labels_agg
chosen_method = "Agglomerative Clustering"

# --- Visualization 1: Original Heatmap ---
plt.figure(figsize=(8, 7))
sns.heatmap(co_occurrence_matrix, annot=True, fmt=".0f", cmap="viridis",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Original Co-occurrence Matrix")
plt.xlabel("Class")
plt.ylabel("Class")
plt.tight_layout()
plt.show()

# --- Visualization 2: Reordered Heatmap ---
# Get the order of indices based on cluster labels
order = np.argsort(cluster_labels)
# Reorder the matrix and labels
reordered_matrix = co_occurrence_matrix[order][:, order]
reordered_names = [class_names[i] for i in order]

plt.figure(figsize=(8, 7))
sns.heatmap(reordered_matrix, annot=True, fmt=".0f", cmap="viridis",
            xticklabels=reordered_names, yticklabels=reordered_names)
plt.title(f"Co-occurrence Matrix Reordered by {chosen_method} Clusters")
plt.xlabel("Class (Reordered)")
plt.ylabel("Class (Reordered)")
plt.tight_layout()
plt.show()

# Add lines to delineate clusters in the reordered heatmap
cluster_boundaries = np.where(np.diff(cluster_labels[order]))[0] + 1
plt.figure(figsize=(8, 7))
sns.heatmap(reordered_matrix, annot=True, fmt=".0f", cmap="viridis",
            xticklabels=reordered_names, yticklabels=reordered_names)
for boundary in cluster_boundaries:
    plt.axhline(boundary, color='red', linewidth=2)
    plt.axvline(boundary, color='red', linewidth=2)
plt.title(f"Reordered Matrix with Cluster Boundaries ({chosen_method})")
plt.xlabel("Class (Reordered)")
plt.ylabel("Class (Reordered)")
plt.tight_layout()
plt.show()


# --- Visualization 3: Dimensionality Reduction (MDS) ---
# MDS uses the distance matrix
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
# Note: Use normalized_stress='auto' or False for newer sklearn versions if you get warnings.
pos = mds.fit_transform(distance_matrix)

plt.figure(figsize=(8, 7))
scatter = plt.scatter(pos[:, 0], pos[:, 1], c=cluster_labels, cmap='viridis', s=100)
# Add labels to points
for i, name in enumerate(class_names):
    plt.text(pos[i, 0] + 0.02, pos[i, 1] + 0.02, name, fontsize=9)

plt.title(f'MDS Projection of Classes based on Co-occurrence Distance\n(Colored by {chosen_method} Clusters)')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.grid(True, linestyle='--', alpha=0.6)
# Add legend if desired (more useful with more clusters)
plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(cluster_labels), title="Clusters")
plt.tight_layout()
plt.show()


# --- Visualization 4: Dendrogram (for Agglomerative Clustering) ---
# Requires the condensed distance matrix (upper or lower triangle)
condensed_distance = squareform(distance_matrix)

# Calculate linkage matrix
# Choose the same linkage method as used in AgglomerativeClustering
linkage_matrix = linkage(condensed_distance, method='average')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix,
           labels=class_names,
           leaf_rotation=90., # rotates the x axis labels
           leaf_font_size=10.) # font size for the x axis labels
plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.xlabel('Class')
plt.ylabel('Distance (Derived from Co-occurrence)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
