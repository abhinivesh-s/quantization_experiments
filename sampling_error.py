import pandas as pd
import numpy as np
from scipy.stats import norm

def precision_recall_sampling_error_with_others(
    df,
    y_true_col,
    y_pred_col,
    confidence=0.95,
    sample_threshold=30,
    exclude_classes=None
):
    """
    Computes per-class precision, recall, SE, and sampling error.
    Groups low-sample classes into 'others' and optionally excludes specific classes.

    Parameters:
    - df (pd.DataFrame): Input data with true and predicted labels.
    - y_true_col (str): Column name for ground truth.
    - y_pred_col (str): Column name for predicted labels.
    - confidence (float): Confidence level for z-score.
    - sample_threshold (int): Minimum number of samples to keep class separate.
    - exclude_classes (list): List of classes to exclude from results (optional).

    Returns:
    - pd.DataFrame with per-class stats and an 'others' row (if applicable).
    """
    if exclude_classes is None:
        exclude_classes = []

    y_true = df[y_true_col]
    y_pred = df[y_pred_col]

    # Include all labels from y_true and y_pred
    all_classes = sorted(np.unique(np.concatenate((y_true.unique(), y_pred.unique()))))

    z = norm.ppf(1 - (1 - confidence) / 2)
    raw_results = []

    for cls in all_classes:
        if cls in exclude_classes:
            continue

        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()
        actual_n = tp + fn

        recall = tp / actual_n if actual_n > 0 else np.nan
        recall_se = np.sqrt(recall * (1 - recall) / actual_n) if actual_n > 0 else np.nan
        recall_sampling_error = z * recall_se if actual_n > 0 else np.nan

        fp = ((y_true != cls) & (y_pred == cls)).sum()
        predicted_n = tp + fp

        precision = tp / predicted_n if predicted_n > 0 else np.nan
        precision_se = np.sqrt(precision * (1 - precision) / predicted_n) if predicted_n > 0 else np.nan
        precision_sampling_error = z * precision_se if predicted_n > 0 else np.nan

        raw_results.append({
            'class': cls,
            'recall': recall,
            'recall_standard_error': recall_se,
            'recall_sampling_error': recall_sampling_error,
            'recall_n': actual_n,
            'precision': precision,
            'precision_standard_error': precision_se,
            'precision_sampling_error': precision_sampling_error,
            'precision_n': predicted_n
        })

    df_raw = pd.DataFrame(raw_results)

    # Determine which to keep vs group into "others"
    main_mask = (df_raw['recall_n'] >= sample_threshold) & (df_raw['precision_n'] >= sample_threshold)
    df_main = df_raw[main_mask].copy()
    df_others = df_raw[~main_mask].copy()

    # Print the 'others' classes
    if not df_others.empty:
        others_classes = df_others['class'].tolist()
        print(f"Classes grouped into 'others': {others_classes}")
        
        others_row = {
            'class': 'others',
            'recall': df_others['recall'].mean(),
            'recall_standard_error': df_others['recall_standard_error'].mean(),
            'recall_sampling_error': z * df_others['recall_standard_error'].mean(),
            'recall_n': df_others['recall_n'].sum(),
            'precision': df_others['precision'].mean(),
            'precision_standard_error': df_others['precision_standard_error'].mean(),
            'precision_sampling_error': z * df_others['precision_standard_error'].mean(),
            'precision_n': df_others['precision_n'].sum()
        }
        df_main = pd.concat([df_main, pd.DataFrame([others_row])], ignore_index=True)

    return df_main.reset_index(drop=True)



df_result = precision_recall_sampling_error_with_others(
    df,
    y_true_col='true_label',
    y_pred_col='predicted_label',
    confidence=0.95,
    sample_threshold=30,
    exclude_classes=['junk', 'unclassified']
)
