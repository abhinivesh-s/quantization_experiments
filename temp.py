import pandas as pd
import numpy as np

# Sample DataFrame with predictions, ground truth, and confidence scores
df = pd.DataFrame({
    'prediction': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'ground_truth': ['A', 'B', 'C', 'C', 'B', 'A', 'A', 'C', 'C', 'A'],
    'confidence': [0.9, 0.8, 0.6, 0.7, 0.95, 0.5, 0.85, 0.4, 0.88, 0.92]
})

# Define confidence thresholds to test
thresholds = np.linspace(0, 1, 11)  # Thresholds from 0.0 to 1.0 (step of 0.1)

# Get unique class labels
classes = df['ground_truth'].unique()

# Store results
metrics_results = {t: {'accuracy': {}, 'precision': {}, 'recall': {}, 'coverage': None} for t in thresholds}

# Loop through thresholds
for threshold in thresholds:
    filtered_df = df[df['confidence'] >= threshold]
    
    # Compute Coverage
    coverage = len(filtered_df) / len(df)
    metrics_results[threshold]['coverage'] = coverage

    # Compute metrics for each class
    for cls in classes:
        # True Positives (TP): Correct predictions for this class
        TP = ((filtered_df['prediction'] == cls) & (filtered_df['ground_truth'] == cls)).sum()
        # False Positives (FP): Predicted as this class but incorrect
        FP = ((filtered_df['prediction'] == cls) & (filtered_df['ground_truth'] != cls)).sum()
        # False Negatives (FN): This class was the actual label but predicted as something else
        FN = ((filtered_df['prediction'] != cls) & (filtered_df['ground_truth'] == cls)).sum()
        # Total samples for this class
        total_cls_samples = (filtered_df['ground_truth'] == cls).sum()

        # Compute Accuracy, Precision, Recall (handle division by zero)
        accuracy = TP / total_cls_samples if total_cls_samples > 0 else None
        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall = TP / (TP + FN) if (TP + FN) > 0 else None

        # Store results
        metrics_results[threshold]['accuracy'][cls] = accuracy
        metrics_results[threshold]['precision'][cls] = precision
        metrics_results[threshold]['recall'][cls] = recall

# Convert to DataFrames
accuracy_df = pd.DataFrame(metrics_results).T['accuracy']
precision_df = pd.DataFrame(metrics_results).T['precision']
recall_df = pd.DataFrame(metrics_results).T['recall']
coverage_df = pd.DataFrame({'coverage': {t: metrics_results[t]['coverage'] for t in thresholds}})

# Display results
print("Accuracy:\n", accuracy_df)
print("\nPrecision:\n", precision_df)
print("\nRecall:\n", recall_df)
print("\nCoverage:\n", coverage_df)
