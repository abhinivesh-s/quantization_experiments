# Step 3: Define a Threshold for Strong Associations
threshold = 5  # Adjust based on dataset size
strong_pairs = {pair for pair, count in co_occurrence_counts.items() if count >= threshold}

# Step 4: Group Classes Based on Shared Co-Occurrences
groups = []
visited = set()

for pair in strong_pairs:
    if pair[0] in visited and pair[1] in visited:
        continue  # Skip if both already assigned

    group = set(pair)
    for other_pair in strong_pairs:
        if group & set(other_pair):  # If there's an overlap, merge
            group.update(other_pair)

    if not any(group <= g for g in groups):  # Avoid duplicate subsets
        groups.append(group)
        visited.update(group)

# Step 5: Print the Grouped Classes
print("\nClass Groups Based on Co-Occurrence:")
for i, group in enumerate(groups, 1):
    print(f"Group {i}: {[class_names[i] for i in group]}")
