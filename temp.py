import pandas as pd
import numpy as np

# Sample DataFrame with predictions, ground truth, and confidence scores
df = pd.DataFrame({
    'prediction': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'ground_truth': ['A', 'B', 'C', 'C', 'B', 'A', 'A', 'C', 'C', 'A'],
    'confidence': [0.9, 0.8, 0.6, 0.7, 0.95, 0.5, 0.85, 0.4, 0.88, 0.92]
})

# Define thresholds (steps of 0.5)
thresholds = np.arange(0, 1.05, 0.5)  # [0.0, 0.5, 1.0]

# Get unique class labels
classes = df['ground_truth'].unique()

# Store results
accuracy_results = []
precision_results = []
recall_results = []
coverage_results = []

for threshold in thresholds:
    filtered_df = df[df['confidence'] >= threshold]

    # Compute overall coverage
    overall_coverage = len(filtered_df) / len(df)

    # Store row results
    accuracy_row = {'Threshold': threshold}
    precision_row = {'Threshold': threshold}
    recall_row = {'Threshold': threshold}
    coverage_row = {'Threshold': threshold, 'Overall': overall_coverage}

    # Compute metrics for each class
    for cls in classes:
        TP = ((filtered_df['prediction'] == cls) & (filtered_df['ground_truth'] == cls)).sum()
        FP = ((filtered_df['prediction'] == cls) & (filtered_df['ground_truth'] != cls)).sum()
        FN = ((filtered_df['prediction'] != cls) & (filtered_df['ground_truth'] == cls)).sum()
        total_cls_samples = (df['ground_truth'] == cls).sum()
        covered_cls_samples = (filtered_df['ground_truth'] == cls).sum()

        # Compute Accuracy, Precision, Recall, Coverage
        accuracy = TP / covered_cls_samples if covered_cls_samples > 0 else None
        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall = TP / (TP + FN) if (TP + FN) > 0 else None
        coverage = covered_cls_samples / total_cls_samples if total_cls_samples > 0 else None

        accuracy_row[cls] = accuracy
        precision_row[cls] = precision
        recall_row[cls] = recall
        coverage_row[cls] = coverage

    # Append rows to results list
    accuracy_results.append(accuracy_row)
    precision_results.append(precision_row)
    recall_results.append(recall_row)
    coverage_results.append(coverage_row)

# Convert to DataFrames
accuracy_df = pd.DataFrame(accuracy_results).set_index("Threshold")
precision_df = pd.DataFrame(precision_results).set_index("Threshold")
recall_df = pd.DataFrame(recall_results).set_index("Threshold")
coverage_df = pd.DataFrame(coverage_results).set_index("Threshold")

# Save DataFrames to Excel
output_file = "classification_metrics.xlsx"
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    accuracy_df.to_excel(writer, sheet_name="Accuracy")
    precision_df.to_excel(writer, sheet_name="Precision")
    recall_df.to_excel(writer, sheet_name="Recall")
    coverage_df.to_excel(writer, sheet_name="Coverage")

print(f"Saved results to {output_file}")
