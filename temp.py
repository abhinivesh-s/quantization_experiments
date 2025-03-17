# Initialize result storage
class_specific_metrics = {
    "Accuracy": {},
    "Precision": {},
    "Recall": {}
}

# Iterate over each class and apply its threshold
for cls in class_thresholds.keys():
    if cls == "Overall":  # Skip overall, as it's not a class
        continue

    class_thresh = class_thresholds[cls]  # Get class-specific threshold

    # Filter data at the class-specific threshold
    df_cls = df[df["confidence"] >= class_thresh]

    # True Positives (TP)
    TP = len(df_cls[(df_cls["prediction"] == cls) & (df_cls["ground_truth"] == cls)])

    # False Positives (FP)
    FP = len(df_cls[(df_cls["prediction"] == cls) & (df_cls["ground_truth"] != cls)])

    # False Negatives (FN)
    FN = len(df[(df["ground_truth"] == cls) & (df["prediction"] != cls)])

    # Compute metrics
    accuracy = TP / len(df_cls) if len(df_cls) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Store results
    class_specific_metrics["Accuracy"][cls] = accuracy
    class_specific_metrics["Precision"][cls] = precision
    class_specific_metrics["Recall"][cls] = recall

# Convert results to DataFrame
class_specific_metrics_df = pd.DataFrame(class_specific_metrics)

# Save to Excel
class_specific_metrics_df.to_excel("class_specific_metrics.xlsx", sheet_name="Class-Specific KPIs")

print("Class-specific KPIs saved to Excel!")
