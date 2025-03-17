# Initialize counters
total_TP = 0
total_FP = 0
total_FN = 0
total_samples = len(df)  # Total number of samples

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

    # Accumulate values
    total_TP += TP
    total_FP += FP
    total_FN += FN

# Compute overall metrics
overall_accuracy = total_TP / total_samples  # Accuracy across all predictions
overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
