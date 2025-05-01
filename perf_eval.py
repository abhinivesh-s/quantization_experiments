import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
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

def evaluate_precomputed_predictions(
    datasets: Dict[str, pd.DataFrame],
    text_col: str, # Still needed for context, maybe error analysis later? Or remove if truly unused.
    target_col: str,
    pred_col: str,
    proba_col: str, # Column with the confidence score (e.g., max probability)
    metadata_cols: Optional[List[str]] = None,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    confidence_thresholds: np.ndarray = np.linspace(0.1, 0.95, 18), # Example thresholds
    plot_charts: bool = True,
    figsize: tuple = (12, 6),
    cm_figsize_scale: float = 0.5, # Scaling factor for CM figure size based on num classes
    max_cm_size: int = 25 # Max dimension for CM plot
):
    """
    Performs comprehensive performance testing using pre-computed predictions
    and probabilities stored in the input DataFrames.

    Args:
        datasets (Dict[str, pd.DataFrame]): A dictionary where keys are descriptive
               names (e.g., 'validation', 'test_set_A') and values are the
               pandas DataFrames containing true labels, predictions, and probabilities.
        text_col (str): Name of the column containing the original text data (optional,
                       can be useful for potential future error analysis linkage).
        target_col (str): Name of the column containing the true labels.
        pred_col (str): Name of the column containing the pre-computed predicted labels.
        proba_col (str): Name of the column containing the pre-computed confidence score
                         (e.g., the maximum probability from the model).
        metadata_cols (Optional[List[str]], optional): List of metadata column names
                       for bucketed analysis. Defaults to None.
        comparisons (Optional[List[Tuple[str, str]]], optional): A list of tuples,
                     where each tuple contains two keys from the `datasets` dict,
                     indicating which dataset results to compare directly.
                     Defaults to None (no direct comparisons printed).
        confidence_thresholds (np.ndarray, optional): Array of confidence
                                         thresholds to evaluate using the `proba_col`.
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
                  'class_names': List of unique true class names found in this dataset.
                  'y_true': True labels array (Optional, for external use)
                  'y_pred': Predicted labels array (Optional)
                  'y_confidence': Confidence score array (Optional)
              }
    """
    print("--- Starting Performance Evaluation from Pre-computed Predictions ---")
    if not datasets:
        print("No datasets provided for evaluation. Exiting.")
        return {}

    results = {}
    all_class_names_ref = None # Store class names from the first dataset for consistency checks

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

        # Check for required columns
        required_cols = [target_col, pred_col, proba_col]
        # Optionally check for text_col if needed downstream, otherwise ignore if missing
        # if text_col not in df.columns:
        #     print(f"   Warning: Text column '{text_col}' not found in dataset '{df_name}'.")
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Skipping dataset '{df_name}': Missing required columns: {missing_cols}.")
            continue

        # Create result structure for this dataset
        results[df_name] = {
            'overall': {},
            'by_metadata': {},
            'by_confidence': pd.DataFrame(),
            'class_names': None,
            'y_true': None,
            'y_pred': None,
            'y_confidence': None
        }

        try:
            print(f"   Dataset '{df_name}' ({len(df)} samples)")
            df_eval = df.copy() # Work on a copy

            # --- 1a. Get Labels, Predictions, and Confidence from DataFrame ---
            print("   Reading pre-computed labels, predictions, and probabilities...")
            y_true = df_eval[target_col]
            y_pred = df_eval[pred_col]
            y_confidence = df_eval[proba_col]

            # Validate probability column seems reasonable (e.g., between 0 and 1)
            if not (y_confidence.min() >= 0 and y_confidence.max() <= 1):
                 print(f"   Warning: Values in probability column '{proba_col}' for dataset '{df_name}' fall outside [0, 1]. Min: {y_confidence.min():.4f}, Max: {y_confidence.max():.4f}. Ensure this column contains valid confidence scores/probabilities.")
            if y_confidence.isnull().any():
                 print(f"   Warning: Probability column '{proba_col}' for dataset '{df_name}' contains NaN values. Results involving confidence might be affected.")
                 # Option: fillna or dropna depending on desired behavior
                 # y_confidence = y_confidence.fillna(0) # Example: fill with 0


            results[df_name]['y_true'] = y_true
            results[df_name]['y_pred'] = y_pred
            results[df_name]['y_confidence'] = y_confidence

            # Determine and store class names based on true labels, check for consistency
            # Using unique true labels ensures classification report targets match ground truth
            current_class_names = sorted(y_true.astype(str).unique()) # Ensure string type for consistency
            results[df_name]['class_names'] = current_class_names
            if all_class_names_ref is None:
                all_class_names_ref = current_class_names
                print(f"   Reference class names set from '{df_name}': {all_class_names_ref}")
            elif not np.array_equal(all_class_names_ref, current_class_names):
                 print(f"   Warning: Class names based on true labels in dataset '{df_name}' {current_class_names} "
                       f"differ from the first dataset's classes {all_class_names_ref}. "
                       "Reports and comparisons might show differing class sets.")

            # --- 1b. Overall Performance ---
            print("   Calculating Overall Performance...")
            metrics = results[df_name]['overall']
            metrics['num_samples'] = len(y_true)
            try:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)

                print(f"     Accuracy:         {metrics['accuracy']:.4f}")
                print(f"     Macro Precision:  {metrics['macro_precision']:.4f} (KPI)")
                print(f"     Macro Recall:     {metrics['macro_recall']:.4f}")
                print(f"     Macro F1:         {metrics['macro_f1']:.4f}")
            except Exception as e:
                print(f"     Error calculating overall metrics for '{df_name}': {e}")
                # Assign NaN or skip if calculation fails
                for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1']:
                    metrics[k] = np.nan


            # --- 1c. Performance by Metadata Columns ---
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
                        grouped = df_eval.groupby(col, observed=False) # Use observed=False for categoricals
                    else: # Try binning continuous variables
                         try:
                             binned_col_name = f'{col}_binned'
                             if pd.api.types.is_datetime64_any_dtype(df_eval[col]):
                                 df_eval[binned_col_name] = pd.cut(df_eval[col], bins=5)
                             else:
                                 try:
                                      df_eval[binned_col_name] = pd.qcut(df_eval[col].astype(float), q=5, duplicates='drop')
                                 except (ValueError, TypeError):
                                      df_eval[binned_col_name] = pd.cut(df_eval[col].astype(float), bins=5)

                             grouped = df_eval.groupby(binned_col_name, observed=False)
                             print(f"       (Binned '{col}' into 5 bins for analysis)")
                         except Exception as e:
                             print(f"       Warning: Could not bin continuous column '{col}'. Skipping. Error: {e}")
                             continue

                    for grp_name, group in grouped:
                        grp_name_str = str(grp_name)
                        group_metrics = {'num_samples': len(group)}
                        if len(group) > 0:
                             grp_y_true = group[target_col]
                             grp_y_pred = group[pred_col]
                             # Use classes present in the specific group for metrics, or overall classes?
                             # Using overall classes (current_class_names) provides consistency but might yield many zeros/NaNs for small groups.
                             # Using group-specific classes might be more informative for the group itself. Let's stick to overall for now for consistency.
                             group_classes = sorted(grp_y_true.astype(str).unique())

                             group_metrics['accuracy'] = accuracy_score(grp_y_true, grp_y_pred)
                             if len(group) >= 2 and grp_y_true.nunique() >= 2:
                                 try:
                                     # Use labels=current_class_names ensures metrics are comparable across groups
                                     group_metrics['macro_precision'] = precision_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                                     group_metrics['macro_recall'] = recall_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                                     group_metrics['macro_f1'] = f1_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                                 except Exception as e_grp:
                                     print(f"         Error calculating macro metrics for group '{grp_name_str}': {e_grp}")
                                     group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan
                             else:
                                 # print(f"       Skipping macro metrics for group '{grp_name_str}' (size={len(group)}, unique_classes={grp_y_true.nunique()})")
                                 group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan
                        else:
                             group_metrics['accuracy'] = np.nan
                             group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan

                        results[df_name]['by_metadata'][col][grp_name_str] = group_metrics

            # --- 1d. Performance by Confidence Threshold ---
            print("   Calculating Performance by Confidence Threshold...")
            confidence_results_list = []
            total_samples = len(y_true)

            for threshold in confidence_thresholds:
                # Ensure y_confidence does not have NaNs for mask operation
                mask = y_confidence.fillna(-1) >= threshold # Fill NaNs with value < 0
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

                    # Calculate metrics using the overall class list for consistency
                    thresh_metrics['accuracy'] = accuracy_score(y_true_thresh, y_pred_thresh)
                    if y_true_thresh.nunique() > 1:
                        try:
                            thresh_metrics['macro_precision'] = precision_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                            thresh_metrics['macro_recall'] = recall_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                            thresh_metrics['macro_f1'] = f1_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                        except Exception as e_thresh:
                             print(f"         Error calculating macro metrics for threshold {threshold:.2f}: {e_thresh}")
                             thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan
                    else:
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
            print("   Generating Classification Report...")
            try:
                # Use current_class_names derived from y_true for target_names
                report = classification_report(y_true, y_pred, target_names=current_class_names, zero_division=0, labels=current_class_names)
                print(report)
                results[df_name]['overall']['classification_report'] = report
            except Exception as e:
                print(f"     Error generating classification report for '{df_name}': {e}")
                results[df_name]['overall']['classification_report'] = "Error generating report."

            # --- 1f. Plotting (optional, per dataset) ---
            if plot_charts:
                 print(f"   Plotting Confusion Matrix for '{df_name}'...")
                 try:
                    num_classes = len(current_class_names)
                    cm_size = min(max(6, num_classes * cm_figsize_scale), max_cm_size)
                    fig, ax = plt.subplots(figsize=(cm_size, cm_size))
                    # Ensure labels arg matches the class names used for consistency
                    cm = confusion_matrix(y_true, y_pred, labels=current_class_names)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_class_names)
                    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
                    plt.title(f"'{df_name}' - Confusion Matrix")
                    plt.tight_layout()
                    plt.show()
                 except Exception as e:
                    print(f"     Error plotting confusion matrix for {df_name}: {e}")

                 print(f"   Plotting Calibration Curve for '{df_name}'...")
                 try:
                    # Ensure y_confidence does not have NaNs before passing to calibration_curve
                    valid_conf_mask = ~y_confidence.isnull()
                    if valid_conf_mask.sum() > 0:
                        prob_true, prob_pred = calibration_curve(
                            (y_true == y_pred)[valid_conf_mask], # Compare true vs pred only for valid confidences
                            y_confidence[valid_conf_mask],      # Use only valid confidences
                            n_bins=10, strategy='uniform'
                        )

                        fig, ax = plt.subplots(figsize=figsize)
                        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, linestyle='-', label=f'{df_name} Confidence Calibration')
                        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
                        ax.set_xlabel("Mean Predicted Confidence (from '{proba_col}')")
                        ax.set_ylabel("Fraction of Positives (Accuracy within bin)")
                        ax.set_title(f"'{df_name}' - Confidence Calibration Curve")
                        ax.grid(True)
                        ax.legend(loc="lower right")
                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"     Skipping calibration curve for '{df_name}' as all confidence values are NaN.")

                 except ValueError as ve:
                     print(f"     Warning: Could not generate calibration curve for '{df_name}'. Maybe only one class predicted or insufficient data in bins? Error: {ve}")
                 except Exception as e:
                    print(f"     Error plotting calibration curve for '{df_name}': {e}")

            processed_datasets.add(df_name) # Mark as successfully processed

        except Exception as e:
            print(f"!!! An error occurred processing dataset '{df_name}': {e}")
            import traceback
            traceback.print_exc()
            if df_name in results:
                del results[df_name]


    # --- 2. Plot Combined Performance vs Confidence ---
    # (Logic remains the same, plotting data from results dict)
    if plot_charts and processed_datasets:
        print("\n2. Plotting Combined Performance vs. Confidence Threshold...")
        plt.figure(figsize=figsize)
        # ... (plotting code for Macro Precision, Accuracy, Coverage vs Threshold remains the same) ...
        # (Ensure legends use df_name correctly)
        for df_name in processed_datasets:
            if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['macro_precision'], marker='o', linestyle='-', label=f'{df_name} Macro Precision')
        plt.title('Macro Precision (KPI) vs. Confidence Threshold')
        plt.xlabel(f'Confidence Threshold (from {proba_col})')
        plt.ylabel('Macro Precision')
        plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout(); plt.show()

        plt.figure(figsize=figsize)
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['accuracy'], marker='o', linestyle='-', label=f'{df_name} Accuracy')
        plt.title('Accuracy vs. Confidence Threshold')
        plt.xlabel(f'Confidence Threshold (from {proba_col})'); plt.ylabel('Accuracy')
        plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout(); plt.show()

        plt.figure(figsize=figsize)
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['coverage'], marker='o', linestyle='-', label=f'{df_name} Coverage')
        plt.title('Coverage vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold (from {proba_col})')
        plt.ylabel('Coverage (Fraction of Samples)'); plt.grid(True); plt.legend(); plt.ylim(0, 1.05); plt.tight_layout(); plt.show()

    elif not processed_datasets:
         print("\n2. Skipping combined confidence plots as no datasets were successfully processed.")
    else:
        print("\n2. Skipping combined confidence plots as plot_charts=False.")


    # --- 3. Perform Specified Comparisons ---
    # (Logic remains the same, comparing overall metrics from results dict)
    print("\n3. Performing Specified Dataset Comparisons...")
    if comparisons and isinstance(comparisons, list):
        # ... (comparison logic remains the same) ...
        compared_pairs = set()
        for comp_pair in comparisons:
            # ...(validation of comp_pair)...
            if not isinstance(comp_pair, tuple) or len(comp_pair) != 2:
                 print(f"   Skipping invalid comparison specification: {comp_pair}. Expected a tuple of two dataset names.")
                 continue
            name1, name2 = comp_pair
            sorted_pair = tuple(sorted((name1, name2)))
            if sorted_pair in compared_pairs: continue
            compared_pairs.add(sorted_pair)
            # ...(check if datasets processed successfully)...
            if name1 not in results or name2 not in results or not results[name1]['overall'] or not results[name2]['overall']:
                 print(f"   Cannot compare '{name1}' vs '{name2}': Results missing for one or both.")
                 continue

            print(f"\n   --- Comparison: '{name1}' vs '{name2}' ---")
            metrics1 = results[name1]['overall']
            metrics2 = results[name2]['overall']
            max_name_len = max(len(name1), len(name2))
            header1 = f"{name1:<{max_name_len}}"
            header2 = f"{name2:<{max_name_len}}"
            print(f"| Metric            | {header1} | {header2} | Change ({name2}-{name1}) |")
            print(f"|-------------------|{'-'*(max_name_len+2)}|{'-'*(max_name_len+2)}|-------------------|")
            kpi_diff = 0.0; kpi_name = 'macro_precision'
            for metric in ['accuracy', kpi_name, 'macro_recall', 'macro_f1']:
                 val1 = metrics1.get(metric, float('nan'))
                 val2 = metrics2.get(metric, float('nan'))
                 change = val2 - val1 if not (np.isnan(val1) or np.isnan(val2)) else float('nan')
                 print(f"| {metric:<17} | {val1: >{max_name_len}.4f} | {val2: >{max_name_len}.4f} | {change: >+17.4f} |")
                 if metric == kpi_name: kpi_diff = change if not np.isnan(change) else 0.0
            print(f"|-------------------|{'-'*(max_name_len+2)}|{'-'*(max_name_len+2)}|-------------------|")
            # ...(print warning/info based on kpi_diff)...
            if kpi_diff < -0.03: print(f"   WARNING: Potential performance degradation detected ({kpi_name}... in '{name2}' vs '{name1}').")
            elif kpi_diff > 0.03: print(f"   INFO: Performance ({kpi_name}) is notably higher in '{name2}' vs '{name1}'.")
            else: print(f"   INFO: Performance ({kpi_name}) is similar between '{name1}' and '{name2}'.")

    elif comparisons: print("   'comparisons' argument provided but is not a list. No comparisons performed.")
    else: print("   No specific comparisons requested.")

    # --- 4. Additional Tests/Suggestions ---
    # (Remains the same, emphasizing error analysis using the provided data)
    print("\n4. Further Analysis Suggestions:")
    print("    - Error Analysis: Manually review examples where prediction in '{pred_col}' != true label in '{target_col}', especially:")
    print("      - High-confidence errors (high value in '{proba_col}').")
    print("      - Errors concentrated in specific metadata groups (check 'by_metadata' results).")
    print("      - Confusion pairs common across multiple datasets.")
    print("      - Link back to '{text_col}' if available for qualitative analysis.")
    print("    - Confidence Distribution Analysis: Examine the distribution of '{proba_col}' for correct vs incorrect predictions.")
    print("    - Domain Shift Analysis: If datasets represent different sources/times, check for performance drift.")


    print("\n--- Evaluation Complete ---")
    return results

# --- Example Usage ---

# Assume you have DataFrames like df_val, df_test, df_holdout
# These DataFrames MUST ALREADY contain the following columns:
#   - 'text' (or your text_col name)
#   - 'RCC_Name' (or your target_col name) - True labels
#   - 'predicted_label' (or your pred_col name) - Model's prediction
#   - 'prediction_confidence' (or your proba_col name) - Model's confidence score

# Example dummy data generation (adding the required prediction/probability columns)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load sample data
print("Loading sample data...")
categories = ['sci.med', 'sci.space', 'talk.politics.guns', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=categories)

# Create base DataFrames
data = pd.DataFrame({'text': newsgroups.data, 'RCC': newsgroups.target})
data['file_type'] = np.random.choice(['PDF', 'DOCX', 'TXT', 'EMAIL'], size=len(data))
data['date_added'] = pd.to_datetime(pd.Timestamp('2023-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=len(data)), unit='D'))
target_map = {i: name for i, name in enumerate(newsgroups.target_names)}
data['RCC_Name'] = data['RCC'].map(target_map)

# Split data
df_train, df_temp = train_test_split(data, test_size=0.5, random_state=42, stratify=data['RCC'])
df_val, df_temp2 = train_test_split(df_temp, test_size=0.6, random_state=123, stratify=df_temp['RCC'])
df_test, df_holdout = train_test_split(df_temp2, test_size=0.5, random_state=456, stratify=df_temp2['RCC'])

print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}, Holdout size: {len(df_holdout)}")

# --- !!! Simulate having predictions already !!! ---
# Train a dummy model ONLY to generate predictions for the example
print("\nTraining a dummy model to generate example predictions...")
temp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('clf', CalibratedClassifierCV(LinearSVC(dual="auto", random_state=42, C=0.1), cv=3))
])
temp_pipeline.fit(df_train['text'], df_train['RCC_Name'])
print("Generating example predictions for Val, Test, Holdout sets...")

# Add prediction columns to the example dataframes
for df_example in [df_val, df_test, df_holdout]:
    df_example['predicted_label'] = temp_pipeline.predict(df_example['text'])
    probas = temp_pipeline.predict_proba(df_example['text'])
    df_example['prediction_confidence'] = np.max(probas, axis=1)
    # Add some noise/variation to make it more realistic if needed
    noise = np.random.uniform(-0.05, 0.05, size=len(df_example))
    df_example['prediction_confidence'] = np.clip(df_example['prediction_confidence'] + noise, 0, 1)

print("Example prediction columns ('predicted_label', 'prediction_confidence') added.")
# --- End of simulation ---


# --- Run the evaluation function using pre-computed columns ---

# 1. Define the datasets dictionary (these now have the prediction columns)
datasets_to_evaluate = {
    'Validation': df_val,
    'Test': df_test,
    'Holdout_2024': df_holdout
    # You could add more datasets here if they have the required columns
}

# 2. Define metadata columns
metadata_cols_to_analyze = ['file_type', 'date_added']

# 3. Define comparisons (optional)
comparisons_to_make = [
    ('Validation', 'Test'),
    ('Test', 'Holdout_2024')
]

# 4. Call the function - **No model object is passed**
evaluation_results = evaluate_precomputed_predictions(
    datasets=datasets_to_evaluate,
    text_col='text',             # Original text column
    target_col='RCC_Name',       # True label column
    pred_col='predicted_label',  # <<< Column with predictions
    proba_col='prediction_confidence', # <<< Column with confidence scores
    metadata_cols=metadata_cols_to_analyze,
    comparisons=comparisons_to_make,
    confidence_thresholds=np.arange(0.2, 1.0, 0.1),
    plot_charts=True
)

# Access results as before
# if 'Test' in evaluation_results:
#     print("\nTest Set Overall Macro Precision:", evaluation_results['Test']['overall']['macro_precision'])
