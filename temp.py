# Step 3: Use a Dynamic Threshold (Above the 75th percentile)
co_occurrence_values = np.array(list(co_occurrence_counts.values()))
threshold = np.percentile(co_occurrence_values, 75)  # Top 25% most frequent co-occurrences

# Step 4: Create a Graph with Strongly Connected Pairs
G = nx.Graph()
strong_pairs = {pair for pair, count in co_occurrence_counts.items() if count >= threshold}
G.add_edges_from(strong_pairs)

# Step 5: Find Connected Components (Groups)
groups = [list(component) for component in nx.connected_components(G)]

# Step 6: Print the Grouped Classes
print("\nClass Groups Based on Strong Co-Occurrence:")
for i, group in enumerate(groups, 1):
    print(f"Group {i}: {[class_names[i] for i in group]}")
