# Step 8: Assign positions to the classes in a circular layout
num_classes = len(class_names)
angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

positions = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(num_classes)}

# Step 9: Draw enclosing circles for hierarchical levels
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Function to draw a circle
def draw_circle(x, y, radius, label=None, color='lightblue'):
    circle = Circle((x, y), radius, edgecolor="black", facecolor=color, alpha=0.3, lw=2)
    ax.add_patch(circle)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

# Step 10: Draw circles for each hierarchical cluster
current_positions = positions.copy()
for i, (c1, c2, dist, _) in enumerate(linked):
    x1, y1 = current_positions[int(c1)]
    x2, y2 = current_positions[int(c2)]
    new_x = (x1 + x2) / 2
    new_y = (y1 + y2) / 2
    radius = np.linalg.norm([x2 - x1, y2 - y1]) / 1.5  # Adjust circle size
    draw_circle(new_x, new_y, radius, label=f"Cluster {num_classes + i}")
    current_positions[num_classes + i] = (new_x, new_y)  # Assign new cluster position

# Step 11: Plot original class labels
for i, name in enumerate(class_names):
    x, y = positions[i]
    ax.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold", bbox=dict(facecolor='white', edgecolor='black'))

plt.title("Hierarchical Clustering with Enclosing Circles", fontsize=14)
plt.show()
