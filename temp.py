import pandas as pd
import numpy as np

# Simulated dataset
np.random.seed(42)
df = pd.DataFrame({
    "prediction": np.random.choice(["A", "B", "C"], size=1000),
    "ground_truth": np.random.choice(["A", "B", "C"], size=1000),
    "confidence": np.random.rand(1000)
})

# Define thresholds and class names
thresholds = np.arange(0, 1.05, 0.5)  # [0.0, 0.5, 1.0]
classes = sorted(df["ground_truth"].unique())  # Extract unique class names

# Initialize metric storage
accuracy_results = {cls: [] for cls in classes}
precision_results = {cls: [] for cls in classes}
recall_results = {cls: [] for cls in classes}
coverage_results = {cls: [] for cls in classes}
overall_accuracy, overall_precision, overall_recall, overall_coverage = [], [], [], []

# Compute metrics at each threshold
for t in thresholds:
    df_filtered = df[df["confidence"] >= t]  # Filter by confidence threshold
    total_predictions = len(df_filtered)

    for cls in classes:
        df_cls = df_filtered[df_filtered["ground_truth"] == cls]

        # True Positives (TP): Correct predictions
        TP = len(df_cls[df_cls["prediction"] == cls])
        
        # False Positives (FP): Incorrect predictions for this class
        FP = len(df_filtered[(df_filtered["prediction"] == cls) & (df_filtered["ground_truth"] != cls)])

        # False Negatives (FN): Missed predictions
        FN = len(df_cls[df_cls["prediction"] != cls])

        # Compute metrics
        accuracy = TP / total_predictions if total_predictions > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        coverage = len(df_cls) / len(df[df["ground_truth"] == cls]) if len(df[df["ground_truth"] == cls]) > 0 else 0

        # Store results
        accuracy_results[cls].append(accuracy)
        precision_results[cls].append(precision)
        recall_results[cls].append(recall)
        coverage_results[cls].append(coverage)

    # Compute overall metrics (Mean across classes)
    overall_accuracy.append(np.mean([accuracy_results[cls][-1] for cls in classes]))
    overall_precision.append(np.mean([precision_results[cls][-1] for cls in classes]))
    overall_recall.append(np.mean([recall_results[cls][-1] for cls in classes]))
    overall_coverage.append(np.mean([coverage_results[cls][-1] for cls in classes]))

# Append 'Overall xx' row
accuracy_results["Overall Accuracy"] = overall_accuracy
precision_results["Overall Precision"] = overall_precision
recall_results["Overall Recall"] = overall_recall
coverage_results["Overall Coverage"] = overall_coverage

# Convert to DataFrames
accuracy_df = pd.DataFrame(accuracy_results, index=thresholds).T
precision_df = pd.DataFrame(precision_results, index=thresholds).T
recall_df = pd.DataFrame(recall_results, index=thresholds).T
coverage_df = pd.DataFrame(coverage_results, index=thresholds).T

# Save to Excel
output_file = "classification_metrics.xlsx"
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    accuracy_df.to_excel(writer, sheet_name="Accuracy")
    precision_df.to_excel(writer, sheet_name="Precision")
    recall_df.to_excel(writer, sheet_name="Recall")
    coverage_df.to_excel(writer, sheet_name="Coverage")

print(f"Saved results to {output_file}")
