import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay, # Keep this for CM plot
)
# Import calibration_curve directly
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Suppress UndefinedMetricWarning for cases where a class/group might be missing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential division by zero with empty groups

def evaluate_classification_model_multi(
    model: Any, # Can be Pipeline, etc.
    datasets: Dict[str, pd.DataFrame],
    text_col: str,
    target_col: str,
    metadata_cols: Optional[List[str]] = None,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    confidence_thresholds: np.ndarray = np.linspace(0.1, 0.95, 18), # Example thresholds
    plot_charts: bool = True,
    figsize: tuple = (12, 6),
    cm_figsize_scale: float = 0.5, # Scaling factor for CM figure size based on num classes
    max_cm_size: int = 25 # Max dimension for CM plot
):
    """
    Performs comprehensive performance testing for a trained text classification model
    on multiple datasets and allows specified comparisons.
    Uses matplotlib directly for calibration plots for compatibility with older sklearn.

    Args:
        model: The trained classification model (e.g., a scikit-learn Pipeline).
               Assumes model has predict() and predict_proba() methods.
        datasets (Dict[str, pd.DataFrame]): A dictionary where keys are descriptive
               names (e.g., 'validation', 'test_set_A', 'holdout_2024') and values
               are the pandas DataFrames to evaluate.
        text_col (str): Name of the column containing the text data.
        target_col (str): Name of the column containing the true labels.
        metadata_cols (Optional[List[str]], optional): List of metadata column names
                       for bucketed analysis. Defaults to None.
        comparisons (Optional[List[Tuple[str, str]]], optional): A list of tuples,
                     where each tuple contains two keys from the `datasets` dict,
                     indicating which dataset results to compare directly.
                     E.g., [('validation', 'test_set_A'), ('validation', 'holdout_2024')].
                     Defaults to None (no direct comparisons printed).
        confidence_thresholds (np.ndarray, optional): Array of confidence
                                         thresholds to evaluate.
                                         Defaults to np.linspace(0.1, 0.95, 18).
        plot_charts (bool, optional): Whether to generate and display plots.
                                      Defaults to True.
        figsize (tuple, optional): Default figure size for most plots. Defaults to (12, 6).
        cm_figsize_scale (float, optional): Factor to scale confusion matrix size by
                                          number of classes. Defaults to 0.5.
        max_cm_size (int, optional): Max dimension (width/height) for confusion matrix plot.
                                     Defaults to 25.

    Returns:
        dict: A dictionary containing performance results keyed by dataset name:
              results[dataset_name] = {
                  'overall': Metrics for the entire dataset.
                  'by_metadata': Metrics broken down by metadata column values.
                  'by_confidence': Metrics and coverage at different thresholds (DataFrame).
                  'class_names': List of unique class names found in this dataset.
                  'y_true': True labels array (Optional, for external use)
                  'y_pred': Predicted labels array (Optional)
                  'y_confidence': Confidence score array (Optional)
              }
    """
    print("--- Starting Model Performance Evaluation ---")
    if not datasets:
        print("No datasets provided for evaluation. Exiting.")
        return {}

    results = {}
    all_class_names = None # Store class names from the first dataset for consistency checks

    # --- 1. Process each dataset ---
    print("\n1. Processing Datasets...")
    processed_datasets = set() # Keep track of datasets processed successfully

    for df_name, df in datasets.items():
        print(f"\n--- Processing Dataset: {df_name} ---")
        if df is None or df.empty:
            print(f"Skipping dataset '{df_name}' as it's empty or None.")
            continue
        if not isinstance(df, pd.DataFrame):
            print(f"Skipping dataset '{df_name}' as it is not a pandas DataFrame.")
            continue
        if text_col not in df.columns or target_col not in df.columns:
            print(f"Skipping dataset '{df_name}': Missing required columns ('{text_col}' or '{target_col}').")
            continue

        # Create result structure for this dataset
        results[df_name] = {
            'overall': {},
            'by_metadata': {},
            'by_confidence': pd.DataFrame(), # Initialize as empty DataFrame
            'class_names': None,
            'y_true': None,
            'y_pred': None,
            'y_confidence': None
        }

        try:
            print(f"   Dataset '{df_name}' ({len(df)} samples)")
            df_eval = df.copy() # Work on a copy

            # --- 1a. Get Predictions and Probabilities ---
            print("   Generating predictions and probabilities...")
            y_true = df_eval[target_col]
            X = df_eval[text_col]
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            y_confidence = np.max(y_proba, axis=1)

            df_eval[f'{target_col}_pred'] = y_pred
            df_eval['confidence_score'] = y_confidence

            results[df_name]['y_true'] = y_true
            results[df_name]['y_pred'] = y_pred
            results[df_name]['y_confidence'] = y_confidence

            current_class_names = getattr(model, 'classes_', sorted(y_true.unique()))
            results[df_name]['class_names'] = current_class_names
            if all_class_names is None:
                all_class_names = current_class_names
            elif not np.array_equal(all_class_names, current_class_names):
                 print(f"   Warning: Class names in dataset '{df_name}' {current_class_names} "
                       f"differ from the first dataset's classes {all_class_names}. "
                       "Reports and comparisons might be affected.")

            # --- 1b. Overall Performance ---
            print("   Calculating Overall Performance...")
            metrics = results[df_name]['overall']
            metrics['num_samples'] = len(y_true)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            print(f"     Accuracy:         {metrics['accuracy']:.4f}")
            print(f"     Macro Precision:  {metrics['macro_precision']:.4f} (KPI)")
            print(f"     Macro Recall:     {metrics['macro_recall']:.4f}")
            print(f"     Macro F1:         {metrics['macro_f1']:.4f}")

            # --- 1c. Performance by Metadata Columns ---
            # (This section remains unchanged)
            if metadata_cols:
                print("   Calculating Performance by Metadata Columns...")
                results[df_name]['by_metadata'] = {}
                for col in metadata_cols:
                    if col not in df_eval.columns:
                        print(f"     Warning: Metadata column '{col}' not found in dataset '{df_name}'. Skipping.")
                        continue

                    print(f"     Analyzing by '{col}'...")
                    results[df_name]['by_metadata'][col] = {}
                    is_categorical = df_eval[col].dtype == 'object' or df_eval[col].dtype.name == 'category' or df_eval[col].nunique() < 20

                    if is_categorical:
                        grouped = df_eval.groupby(col)
                    else: # Try binning continuous variables
                         try:
                             binned_col_name = f'{col}_binned'
                             if pd.api.types.is_datetime64_any_dtype(df_eval[col]):
                                 df_eval[binned_col_name] = pd.cut(df_eval[col], bins=5) # Adjust bins as needed
                             else:
                                 try:
                                      df_eval[binned_col_name] = pd.qcut(df_eval[col].astype(float), q=5, duplicates='drop')
                                 except (ValueError, TypeError):
                                      df_eval[binned_col_name] = pd.cut(df_eval[col].astype(float), bins=5) # Force float for cut

                             grouped = df_eval.groupby(binned_col_name)
                             print(f"       (Binned '{col}' into 5 bins for analysis)")
                         except Exception as e:
                             print(f"       Warning: Could not bin continuous column '{col}'. Skipping. Error: {e}")
                             continue

                    for grp_name, group in grouped:
                        grp_name_str = str(grp_name)
                        if len(group) < 2 or group[target_col].nunique() < 2 :
                            results[df_name]['by_metadata'][col][grp_name_str] = {
                                'num_samples': len(group),
                                'accuracy': accuracy_score(group[target_col], group[f'{target_col}_pred']) if len(group)>0 else 0,
                                'macro_precision': np.nan, 'macro_recall': np.nan, 'macro_f1': np.nan
                            }
                            continue

                        group_metrics = {}
                        group_metrics['num_samples'] = len(group)
                        group_metrics['accuracy'] = accuracy_score(group[target_col], group[f'{target_col}_pred'])
                        group_metrics['macro_precision'] = precision_score(group[target_col], group[f'{target_col}_pred'], average='macro', zero_division=0)
                        group_metrics['macro_recall'] = recall_score(group[target_col], group[f'{target_col}_pred'], average='macro', zero_division=0)
                        group_metrics['macro_f1'] = f1_score(group[target_col], group[f'{target_col}_pred'], average='macro', zero_division=0)
                        results[df_name]['by_metadata'][col][grp_name_str] = group_metrics

            # --- 1d. Performance by Confidence Threshold ---
            # (This section remains unchanged)
            print("   Calculating Performance by Confidence Threshold...")
            confidence_results_list = []
            total_samples = len(y_true)

            for threshold in confidence_thresholds:
                mask = y_confidence >= threshold
                covered_samples = np.sum(mask)
                coverage = covered_samples / total_samples if total_samples > 0 else 0

                thresh_metrics = {
                    'threshold': threshold,
                    'coverage': coverage,
                    'num_samples_covered': covered_samples
                }

                if covered_samples > 0:
                    y_true_thresh = y_true[mask]
                    y_pred_thresh = y_pred[mask]

                    if y_true_thresh.nunique() > 1:
                        thresh_metrics['accuracy'] = accuracy_score(y_true_thresh, y_pred_thresh)
                        thresh_metrics['macro_precision'] = precision_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0)
                        thresh_metrics['macro_recall'] = recall_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0)
                        thresh_metrics['macro_f1'] = f1_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0)
                    else:
                        thresh_metrics['accuracy'] = accuracy_score(y_true_thresh, y_pred_thresh)
                        thresh_metrics['macro_precision'] = np.nan
                        thresh_metrics['macro_recall'] = np.nan
                        thresh_metrics['macro_f1'] = np.nan
                else:
                    thresh_metrics['accuracy'] = np.nan
                    thresh_metrics['macro_precision'] = np.nan
                    thresh_metrics['macro_recall'] = np.nan
                    thresh_metrics['macro_f1'] = np.nan

                confidence_results_list.append(thresh_metrics)
            results[df_name]['by_confidence'] = pd.DataFrame(confidence_results_list)
            print(f"     Analyzed {len(confidence_thresholds)} confidence thresholds.")


            # --- 1e. Detailed Classification Report ---
            # (This section remains unchanged)
            print("   Generating Classification Report...")
            report = classification_report(y_true, y_pred, target_names=[str(tn) for tn in current_class_names], zero_division=0)
            print(report)
            results[df_name]['overall']['classification_report'] = report


            # --- 1f. Plotting (optional, per dataset) ---
            if plot_charts:
                 print(f"   Plotting Confusion Matrix for '{df_name}'...")
                 try:
                    num_classes = len(current_class_names)
                    cm_size = min(max(6, num_classes * cm_figsize_scale), max_cm_size)
                    fig, ax = plt.subplots(figsize=(cm_size, cm_size))
                    cm = confusion_matrix(y_true, y_pred, labels=current_class_names)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(tn) for tn in current_class_names])
                    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
                    plt.title(f"'{df_name}' - Confusion Matrix")
                    plt.tight_layout()
                    plt.show()
                 except Exception as e:
                    print(f"     Error plotting confusion matrix for {df_name}: {e}")

                 # --- MODIFIED SECTION: Calibration Plot ---
                 if hasattr(model, 'predict_proba'):
                    print(f"   Plotting Calibration Curve for '{df_name}'...")
                    try:
                        # Calculate calibration curve data
                        # Using y_true == y_pred checks if the prediction was correct for the given confidence
                        prob_true, prob_pred = calibration_curve(y_true == y_pred, y_confidence, n_bins=10, strategy='uniform')

                        # Create plot using matplotlib directly
                        fig, ax = plt.subplots(figsize=figsize)

                        # Plot the calibration curve
                        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, linestyle='-', label=f'{df_name} Confidence Calibration')

                        # Plot the perfectly calibrated line (reference)
                        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')

                        # Add labels, title, grid, legend
                        ax.set_xlabel("Mean Predicted Confidence (Max Probability)")
                        ax.set_ylabel("Fraction of Positives (Accuracy within bin)")
                        ax.set_title(f"'{df_name}' - Confidence Calibration Curve")
                        ax.grid(True)
                        ax.legend(loc="lower right")
                        plt.tight_layout()
                        plt.show()

                    except ValueError as ve:
                         # Handle cases where calibration_curve might fail (e.g., only one class present after filtering)
                         print(f"     Warning: Could not generate calibration curve for '{df_name}'. Maybe only one class predicted? Error: {ve}")
                    except Exception as e:
                        print(f"     Error plotting calibration curve for '{df_name}': {e}")
                 # --- END OF MODIFIED SECTION ---

            processed_datasets.add(df_name) # Mark as successfully processed

        except Exception as e:
            print(f"!!! An error occurred processing dataset '{df_name}': {e}")
            import traceback
            traceback.print_exc()
            if df_name in results:
                del results[df_name]


    # --- 2. Plot Combined Performance vs Confidence ---
    # (This section remains unchanged)
    if plot_charts and processed_datasets:
        print("\n2. Plotting Combined Performance vs. Confidence Threshold...")
        plt.figure(figsize=figsize)
        for df_name in processed_datasets:
            if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['macro_precision'], marker='o', linestyle='-', label=f'{df_name} Macro Precision')
        plt.title('Macro Precision (KPI) vs. Model Confidence Threshold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Macro Precision')
        plt.grid(True)
        plt.legend()
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['accuracy'], marker='o', linestyle='-', label=f'{df_name} Accuracy')
        plt.title('Accuracy vs. Model Confidence Threshold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['coverage'], marker='o', linestyle='-', label=f'{df_name} Coverage')
        plt.title('Coverage vs. Model Confidence Threshold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Coverage (Fraction of Samples)')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.show()
    elif not processed_datasets:
         print("\n2. Skipping combined confidence plots as no datasets were successfully processed.")
    else:
        print("\n2. Skipping combined confidence plots as plot_charts=False.")


    # --- 3. Perform Specified Comparisons ---
    # (This section remains unchanged)
    print("\n3. Performing Specified Dataset Comparisons...")
    if comparisons and isinstance(comparisons, list):
        compared_pairs = set()
        for comp_pair in comparisons:
            if not isinstance(comp_pair, tuple) or len(comp_pair) != 2:
                print(f"   Skipping invalid comparison specification: {comp_pair}. Expected a tuple of two dataset names.")
                continue

            name1, name2 = comp_pair
            sorted_pair = tuple(sorted((name1, name2)))
            if sorted_pair in compared_pairs:
                continue
            compared_pairs.add(sorted_pair)

            if name1 not in results or name2 not in results:
                print(f"   Cannot compare '{name1}' vs '{name2}': One or both datasets were not processed successfully.")
                continue
            if not results[name1]['overall'] or not results[name2]['overall']:
                 print(f"   Cannot compare '{name1}' vs '{name2}': Missing overall results for one or both.")
                 continue

            print(f"\n   --- Comparison: '{name1}' vs '{name2}' ---")
            metrics1 = results[name1]['overall']
            metrics2 = results[name2]['overall']
            max_name_len = max(len(name1), len(name2))
            header1 = f"{name1:<{max_name_len}}"
            header2 = f"{name2:<{max_name_len}}"

            print(f"| Metric            | {header1} | {header2} | Change ({name2}-{name1}) |")
            print(f"|-------------------|{'-'*(max_name_len+2)}|{'-'*(max_name_len+2)}|-------------------|")

            kpi_diff = 0.0
            kpi_name = 'macro_precision'

            for metric in ['accuracy', kpi_name, 'macro_recall', 'macro_f1']:
                 val1 = metrics1.get(metric, float('nan'))
                 val2 = metrics2.get(metric, float('nan'))
                 change = val2 - val1 if not (np.isnan(val1) or np.isnan(val2)) else float('nan')
                 print(f"| {metric:<17} | {val1: >{max_name_len}.4f} | {val2: >{max_name_len}.4f} | {change: >+17.4f} |")
                 if metric == kpi_name:
                     kpi_diff = change if not np.isnan(change) else 0.0

            print(f"|-------------------|{'-'*(max_name_len+2)}|{'-'*(max_name_len+2)}|-------------------|")

            if kpi_diff < -0.03:
                 print(f"   WARNING: Potential performance degradation detected ({kpi_name} decreased significantly in '{name2}' compared to '{name1}').")
            elif kpi_diff > 0.03:
                 print(f"   INFO: Performance ({kpi_name}) is notably higher in '{name2}' compared to '{name1}'.")
            else:
                 print(f"   INFO: Performance ({kpi_name}) is similar between '{name1}' and '{name2}'.")

    elif comparisons:
        print("   'comparisons' argument provided but is not a list. No comparisons performed.")
    else:
        print("   No specific comparisons requested.")


    # --- 4. Additional Tests/Suggestions ---
    # (This section remains unchanged)
    print("\n4. Further Analysis Suggestions:")
    print("    - Error Analysis: Manually review examples where the model was wrong...")
    print("    - Feature Importance Analysis (if possible)...")
    print("    - Latency/Throughput Testing.")
    print("    - Robustness Testing (typos, paraphrasing).")
    print("    - Domain Shift Analysis...")


    print("\n--- Evaluation Complete ---")
    return results


# --- Example Usage (remains the same) ---

# Assume you have:
# model: Your trained Pipeline(TfidfVectorizer(...), CalibratedClassifierCV(LinearSVC(...)))
# df_val: Your validation pandas DataFrame
# df_test: Your test pandas DataFrame
# df_holdout: Another holdout pandas DataFrame

# Example dummy data (Replace with your actual data and model)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups # Example dataset

# Load sample data
print("Loading sample data...")
categories = ['sci.med', 'sci.space', 'talk.politics.guns', 'comp.graphics'] # Using only 4 classes for faster demo
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=categories)

# Create dummy DataFrames
data = pd.DataFrame({'text': newsgroups.data, 'RCC': newsgroups.target})
data['file_type'] = np.random.choice(['PDF', 'DOCX', 'TXT', 'EMAIL'], size=len(data))
data['date_added'] = pd.to_datetime(pd.Timestamp('2023-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=len(data)), unit='D')) # Wider date range
target_map = {i: name for i, name in enumerate(newsgroups.target_names)}
data['RCC_Name'] = data['RCC'].map(target_map)

# Split data
df_train, df_temp = train_test_split(data, test_size=0.5, random_state=42, stratify=data['RCC'])
df_val, df_temp2 = train_test_split(df_temp, test_size=0.6, random_state=123, stratify=df_temp['RCC'])
df_test, df_holdout = train_test_split(df_temp2, test_size=0.5, random_state=456, stratify=df_temp2['RCC'])

print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}, Holdout size: {len(df_holdout)}")

# Define and Train a dummy model
print("\nTraining a dummy model for demonstration...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('clf', CalibratedClassifierCV(LinearSVC(dual="auto", random_state=42, C=0.1), cv=3))
])
pipeline.fit(df_train['text'], df_train['RCC_Name'])
print("Dummy model trained.")

# --- Run the evaluation function ---
datasets_to_evaluate = {
    'Validation': df_val,
    'Test': df_test,
    'Holdout_2024': df_holdout,
    'Test_Only_PDF': df_test[df_test['file_type'] == 'PDF']
}
metadata_cols_to_analyze = ['file_type', 'date_added']
comparisons_to_make = [
    ('Validation', 'Test'),
    ('Test', 'Holdout_2024'),
    ('Test', 'Test_Only_PDF')
]

evaluation_results = evaluate_classification_model_multi(
    model=pipeline,
    datasets=datasets_to_evaluate,
    text_col='text',
    target_col='RCC_Name',
    metadata_cols=metadata_cols_to_analyze,
    comparisons=comparisons_to_make,
    confidence_thresholds=np.arange(0.2, 1.0, 0.1),
    plot_charts=True
)
