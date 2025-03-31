import numpy as np
import matplotlib.pyplot as plt

# 1. Define your bins and generate some sample data
bins = [0, 0.05, 0.1, 1]
# Generate some sample data (replace with your actual data)
np.random.seed(42) # for reproducibility
# Example: 1000 numbers skewed towards 0, within the 0-1 range
data = np.random.power(0.5, 1000) * 0.8 # Generates data mostly < 0.8

# Filter data to be within the overall bin range if necessary
# (not strictly needed here as data is 0-1, but good practice)
data = data[(data >= bins[0]) & (data <= bins[-1])]

# 2. Calculate histogram counts using numpy.histogram
counts, actual_bin_edges = np.histogram(data, bins=bins)

# We have counts for bins: [0, 0.05), [0.05, 0.1), [0.1, 1]
print(f"Bin Edges: {actual_bin_edges}")
print(f"Counts per bin: {counts}")

# 3. Prepare for plotting with equal visual widths
num_bins = len(counts)
# Create x-positions for the bars (0, 1, 2, ...)
x_positions = np.arange(num_bins)

# 4. Create descriptive labels for each bin
bin_labels = []
for i in range(num_bins):
    # Format the label string based on bin edges
    # Using [low, high) format, except for the last bin which is [low, high]
    lower = actual_bin_edges[i]
    upper = actual_bin_edges[i+1]
    if i == num_bins - 1:
         # Last bin includes the upper edge according to numpy.histogram behavior
         label = f"[{lower:.2f} - {upper:.2f}]"
    else:
         label = f"[{lower:.2f} - {upper:.2f})"
    bin_labels.append(label)
    # Alternative simpler label:
    # bin_labels.append(f"{lower:.2f} to {upper:.2f}")

print(f"Bin Labels: {bin_labels}")

# 5. Create the bar plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the bars at the equally spaced positions
# Use width < 1 for some visual spacing between bars
bar_width = 0.8
ax.bar(x_positions, counts, width=bar_width, color='skyblue', edgecolor='black', align='center')

# 6. Set the x-axis ticks and labels
# Position the ticks at the center of the bars (which are at 0, 1, 2...)
ax.set_xticks(x_positions)
# Set the labels for these ticks
ax.set_xticklabels(bin_labels)

# 7. Add labels and title
ax.set_xlabel("Bin Ranges (Displayed with Equal Visual Width)")
ax.set_ylabel("Frequency (Count)")
ax.set_title("Histogram with Custom Bin Visualization")

# Optional: Add count numbers on top of bars
for i, count in enumerate(counts):
    # Add a small offset (e.g., 1% of max count) for the text position
    text_y_offset = counts.max() * 0.01
    ax.text(x_positions[i], count + text_y_offset, str(count), ha='center', va='bottom')

# 8. Display the plot
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()
