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

overall_accuracy_results = []
overall_precision_results = []
overall_recall_results = []

for threshold in thresholds:
    filtered_df = df[df['confidence'] >= threshold]

    # Compute overall coverage
    overall_coverage = len(filtered_df) / len(df) if len(df) > 0 else 0

    # Initialize row dictionaries
    accuracy_row = {'Class': 'Accuracy'}
    precision_row = {'Class': 'Precision'}
    recall_row = {'Class': 'Recall'}
    coverage_row = {'Class': 'Coverage', 'Overall': overall_coverage}

    overall_TP = 0
    overall_FP = 0
    overall_FN = 0
    total_covered_samples = len(filtered_df)

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

        # Store class-wise metrics
        accuracy_row[cls] = accuracy
        precision_row[cls] = precision
        recall_row[cls] = recall
        coverage_row[cls] = coverage

        # Accumulate for overall metrics
        overall_TP += TP
        overall_FP += FP
        overall_FN += FN

    # Compute overall accuracy, precision, and recall
    overall_accuracy = overall_TP / total_covered_samples if total_covered_samples > 0 else None
    overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else None
    overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else None

    overall_accuracy_results.append({'Class': 'Overall Accuracy', threshold: overall_accuracy})
    overall_precision_results.append({'Class': 'Overall Precision', threshold: overall_precision})
    overall_recall_results.append({'Class': 'Overall Recall', threshold: overall_recall})

    # Append rows to results list
    accuracy_results.append(accuracy_row)
    precision_results.append(precision_row)
    recall_results.append(recall_row)
    coverage_results.append(coverage_row)

# Convert to DataFrames and transpose to get classes as index, thresholds as columns
accuracy_df = pd.DataFrame(accuracy_results).set_index("Class").T
precision_df = pd.DataFrame(precision_results).set_index("Class").T
recall_df = pd.DataFrame(recall_results).set_index("Class").T
coverage_df = pd.DataFrame(coverage_results).set_index("Class").T

# Convert overall metrics and merge into corresponding DataFrames
overall_accuracy_df = pd.DataFrame(overall_accuracy_results).set_index("Class").T
overall_precision_df = pd.DataFrame(overall_precision_results).set_index("Class").T
overall_recall_df = pd.DataFrame(overall_recall_results).set_index("Class").T

accuracy_df = pd.concat([accuracy_df, overall_accuracy_df])
precision_df = pd.concat([precision_df, overall_precision_df])
recall_df = pd.concat([recall_df, overall_recall_df])

# Save DataFrames to Excel
output_file = "classification_metrics.xlsx"
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    accuracy_df.to_excel(writer, sheet_name="Accuracy")
    precision_df.to_excel(writer, sheet_name="Precision")
    recall_df.to_excel(writer, sheet_name="Recall")
    coverage_df.to_excel(writer, sheet_name="Coverage")

print(f"Saved results to {output_file}")
