# Step 3: Define a Threshold for Strong Associations
threshold = 5  # Adjust based on dataset size
strong_pairs = {pair for pair, count in co_occurrence_counts.items() if count >= threshold}

# Step 4: Create a Graph and Add Strongly Connected Pairs
G = nx.Graph()
G.add_edges_from(strong_pairs)

# Step 5: Find Connected Components (Groups)
groups = [list(component) for component in nx.connected_components(G)]

# Step 6: Print the Grouped Classes
print("\nClass Groups Based on Co-Occurrence:")
for i, group in enumerate(groups, 1):
    print(f"Group {i}: {[class_names[i] for i in group]}")
