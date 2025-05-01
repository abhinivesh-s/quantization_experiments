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
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pandas.api.types import is_categorical_dtype

# Suppress UndefinedMetricWarning for cases where a class/group might be missing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential division by zero with empty groups

def evaluate_precomputed_predictions(
    datasets: Dict[str, pd.DataFrame],
    text_col: str, # Optional context
    target_col: str,
    pred_col: str,
    proba_col: str,
    metadata_cols: Optional[List[str]] = None,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    confidence_thresholds: np.ndarray = np.arange(0.05, 1.05, 0.05), # Updated default
    plot_charts: bool = True,
    figsize: tuple = (12, 6),
    dpi: int = 100, # Added DPI argument
    cm_figsize_scale: float = 0.5,
    max_cm_size: int = 25
):
    """
    Performs comprehensive performance testing using pre-computed predictions
    and probabilities stored in the input DataFrames. Includes detailed metadata
    breakdown and DPI setting for plots.

    Args:
        datasets (Dict[str, pd.DataFrame]): Dict of dataset names to DataFrames.
        text_col (str): Name of the original text column (optional context).
        target_col (str): Name of the true label column.
        pred_col (str): Name of the predicted label column.
        proba_col (str): Name of the confidence score/probability column.
        metadata_cols (Optional[List[str]], optional): Metadata columns for analysis.
        comparisons (Optional[List[Tuple[str, str]]], optional): Dataset pairs to compare.
        confidence_thresholds (np.ndarray, optional): Confidence thresholds to evaluate.
                                         Defaults to np.arange(0.05, 1.05, 0.05).
        plot_charts (bool, optional): Generate plots. Defaults to True.
        figsize (tuple, optional): Default figure size. Defaults to (12, 6).
        dpi (int, optional): Dots Per Inch for plot resolution. Defaults to 100.
        cm_figsize_scale (float, optional): Confusion matrix size scaling factor. Defaults to 0.5.
        max_cm_size (int, optional): Max confusion matrix dimension. Defaults to 25.

    Returns:
        dict: A dictionary containing performance results keyed by dataset name.
    """
    print("--- Starting Performance Evaluation from Pre-computed Predictions ---")
    if not datasets:
        print("No datasets provided for evaluation. Exiting.")
        return {}

    results = {}
    all_class_names_ref = None

    # --- 1. Process each dataset ---
    print("\n1. Processing Datasets...")
    processed_datasets = set()

    for df_name, df in datasets.items():
        print(f"\n--- Processing Dataset: {df_name} ---")
        # (Input validation checks remain the same)
        if df is None or df.empty: print(f"Skipping dataset '{df_name}' (empty/None)."); continue
        if not isinstance(df, pd.DataFrame): print(f"Skipping dataset '{df_name}' (not DataFrame)."); continue
        required_cols = [target_col, pred_col, proba_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: print(f"Skipping dataset '{df_name}': Missing cols: {missing_cols}."); continue

        results[df_name] = {
            'overall': {}, 'by_metadata': {}, 'by_confidence': pd.DataFrame(),
            'class_names': None, 'y_true': None, 'y_pred': None, 'y_confidence': None
        }

        try:
            print(f"   Dataset '{df_name}' ({len(df)} samples)")
            df_eval = df.copy()

            # --- 1a. Get Labels, Predictions, Confidence ---
            print("   Reading pre-computed data...")
            y_true = df_eval[target_col]
            y_pred = df_eval[pred_col]
            y_confidence = df_eval[proba_col]

            if not (y_confidence.min() >= 0 and y_confidence.max() <= 1): print(f"   Warning: Probabilities in '{proba_col}' outside [0, 1] for '{df_name}'.")
            if y_confidence.isnull().any(): print(f"   Warning: NaNs found in '{proba_col}' for '{df_name}'.")

            results[df_name]['y_true'] = y_true
            results[df_name]['y_pred'] = y_pred
            results[df_name]['y_confidence'] = y_confidence

            current_class_names = sorted(y_true.astype(str).unique())
            results[df_name]['class_names'] = current_class_names
            if all_class_names_ref is None: all_class_names_ref = current_class_names; print(f"   Reference classes: {all_class_names_ref}")
            elif not np.array_equal(all_class_names_ref, current_class_names): print(f"   Warning: True classes differ: {current_class_names} vs {all_class_names_ref}")

            # --- 1b. Overall Performance ---
            print("   Calculating Overall Performance...")
            # (Calculation logic remains the same, using labels=current_class_names)
            metrics = results[df_name]['overall']
            metrics['num_samples'] = len(y_true)
            try:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                # Ensure consistent labeling for metrics
                metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                # ... weighted metrics ...
                metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                print(f"     Accuracy:         {metrics['accuracy']:.4f}")
                print(f"     Macro Precision:  {metrics['macro_precision']:.4f} (KPI)")
                print(f"     Macro Recall:     {metrics['macro_recall']:.4f}")
                print(f"     Macro F1:         {metrics['macro_f1']:.4f}")
            except Exception as e:
                print(f"     Error calculating overall metrics for '{df_name}': {e}")
                for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1']: metrics[k] = np.nan

            # --- 1c. Performance by Metadata Columns ---
            if metadata_cols:
                print("\n   Calculating Performance by Metadata Columns...")
                results[df_name]['by_metadata'] = {}
                for col in metadata_cols:
                    if col not in df_eval.columns:
                        print(f"     Warning: Metadata column '{col}' not found in dataset '{df_name}'. Skipping.")
                        continue

                    print(f"     ----- Analyzing by Metadata Column: '{col}' -----")
                    metadata_results_list = [] # To store results for table printing
                    results[df_name]['by_metadata'][col] = {}

                    # Grouping logic (Categorical vs Continuous Binning)
                    is_explicitly_categorical = is_categorical_dtype(df_eval[col])
                    is_likely_categorical = df_eval[col].dtype == 'object' or df_eval[col].nunique() < 20 # Heuristic

                    if is_explicitly_categorical or is_likely_categorical:
                        print(f"       Treating '{col}' as categorical.")
                        # Convert to category if it's object type for better groupby handling
                        if df_eval[col].dtype == 'object':
                           try:
                               df_eval[col] = df_eval[col].astype('category')
                           except Exception: # Handle potential mixed types
                               print(f"       Warning: Could not convert '{col}' to category dtype.")
                        grouped = df_eval.groupby(col, observed=False) # observed=False includes all categories
                        binned = False
                    else: # Try binning continuous variables
                         print(f"       Treating '{col}' as continuous, attempting binning.")
                         try:
                             binned_col_name = f'{col}_binned'
                             if pd.api.types.is_datetime64_any_dtype(df_eval[col]):
                                 df_eval[binned_col_name] = pd.cut(df_eval[col], bins=5)
                             else: # Numeric or other - try qcut first
                                 try: # qcut preferred for potentially skewed numeric data
                                      df_eval[binned_col_name] = pd.qcut(df_eval[col].astype(float), q=5, duplicates='drop')
                                      print(f"       (Binned '{col}' into 5 quantiles)")
                                 except (ValueError, TypeError): # Fallback to equi-width cut
                                      df_eval[binned_col_name] = pd.cut(df_eval[col].astype(float), bins=5)
                                      print(f"       (Binned '{col}' into 5 equal-width intervals)")

                             grouped = df_eval.groupby(binned_col_name, observed=False)
                             binned = True
                         except Exception as e:
                             print(f"       Warning: Could not bin continuous column '{col}'. Skipping analysis for this column. Error: {e}")
                             results[df_name]['by_metadata'][col] = {"ERROR": f"Binning failed: {e}"}
                             continue # Skip to next metadata column


                    # Calculate metrics for each group
                    for grp_name, group in grouped:
                        grp_name_str = str(grp_name) # Ensure string for dict key and printing
                        group_metrics = {'num_samples': len(group)}
                        if len(group) > 0:
                             grp_y_true = group[target_col]
                             grp_y_pred = group[pred_col]
                             group_classes = sorted(grp_y_true.astype(str).unique())

                             group_metrics['accuracy'] = accuracy_score(grp_y_true, grp_y_pred)

                             # Calculate macro metrics only if feasible
                             if len(group) >= 2 and grp_y_true.nunique() >= 2:
                                 try:
                                     # Use overall class list for consistent comparison across groups
                                     group_metrics['macro_precision'] = precision_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                                     group_metrics['macro_recall'] = recall_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                                     group_metrics['macro_f1'] = f1_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                                 except Exception as e_grp:
                                     # print(f"         Error calculating macro metrics for group '{grp_name_str}': {e_grp}") # Optional verbose logging
                                     group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan
                             else: # Not enough data/classes for macro averages in this specific group
                                 group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan
                        else: # Empty group (can happen with observed=False)
                             group_metrics['accuracy'] = np.nan
                             group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan

                        results[df_name]['by_metadata'][col][grp_name_str] = group_metrics
                        # Add group name to the dict for table printing
                        group_metrics['group_name'] = grp_name_str
                        metadata_results_list.append(group_metrics)

                    # --- Print Metadata Summary Table ---
                    if metadata_results_list:
                        meta_df = pd.DataFrame(metadata_results_list)
                        # Determine max length of group names for formatting
                        max_grp_len = meta_df['group_name'].astype(str).map(len).max()
                        max_grp_len = max(max_grp_len, len(col)) # Ensure header fits

                        print(f"       Summary for '{col}':")
                        header = f"| {col + ' Group':<{max_grp_len}} | Samples | Accuracy | Macro Precision |"
                        print(header)
                        print(f"|{'-'*(max_grp_len+2)}|---------|----------|-----------------|")
                        for _, row in meta_df.iterrows():
                             print(f"| {str(row['group_name']):<{max_grp_len}} | {row['num_samples']:>7d} | {row.get('accuracy', float('nan')):>8.4f} | {row.get('macro_precision', float('nan')):>15.4f} |")
                        print(f"|{'-'*(max_grp_len+2)}|---------|----------|-----------------|")
                    else:
                        print(f"       No groups found or processed for '{col}'.")
                    print(f"     ----- Finished Analyzing: '{col}' -----")


            # --- 1d. Performance by Confidence Threshold ---
            print("\n   Calculating Performance by Confidence Threshold...")
            # (Calculation logic remains the same, using labels=current_class_names)
            confidence_results_list = []
            total_samples = len(y_true)
            y_confidence_filled = y_confidence.fillna(-1) # Handle potential NaNs for comparison

            for threshold in confidence_thresholds:
                mask = y_confidence_filled >= threshold
                covered_samples = np.sum(mask)
                coverage = covered_samples / total_samples if total_samples > 0 else 0
                thresh_metrics = {'threshold': threshold, 'coverage': coverage, 'num_samples_covered': covered_samples}

                if covered_samples > 0:
                    y_true_thresh = y_true[mask]
                    y_pred_thresh = y_pred[mask]
                    thresh_metrics['accuracy'] = accuracy_score(y_true_thresh, y_pred_thresh)
                    if y_true_thresh.nunique() > 1:
                        try:
                            thresh_metrics['macro_precision'] = precision_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                            thresh_metrics['macro_recall'] = recall_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                            thresh_metrics['macro_f1'] = f1_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                        except Exception: # Catch potential errors if subset causes issues
                             thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan
                    else: thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan
                else: thresh_metrics['accuracy'], thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan, np.nan
                confidence_results_list.append(thresh_metrics)

            results[df_name]['by_confidence'] = pd.DataFrame(confidence_results_list)
            print(f"     Analyzed {len(confidence_thresholds)} confidence thresholds.")


            # --- 1e. Detailed Classification Report ---
            print("\n   Generating Classification Report...")
            # (Logic remains the same, using labels=current_class_names)
            try:
                report = classification_report(y_true, y_pred, target_names=current_class_names, zero_division=0, labels=current_class_names)
                print(report)
                results[df_name]['overall']['classification_report'] = report
            except Exception as e:
                print(f"     Error generating classification report for '{df_name}': {e}")
                results[df_name]['overall']['classification_report'] = "Error generating report."


            # --- 1f. Plotting (Applying DPI) ---
            if plot_charts:
                 print(f"\n   Plotting Charts for '{df_name}' (DPI={dpi})...")
                 print(f"     Plotting Confusion Matrix...")
                 try:
                    num_classes = len(current_class_names)
                    cm_size = min(max(6, num_classes * cm_figsize_scale), max_cm_size)
                    # Apply DPI here
                    fig, ax = plt.subplots(figsize=(cm_size, cm_size), dpi=dpi)
                    cm = confusion_matrix(y_true, y_pred, labels=current_class_names)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_class_names)
                    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
                    plt.title(f"'{df_name}' - Confusion Matrix")
                    plt.tight_layout()
                    plt.show()
                 except Exception as e:
                    print(f"     Error plotting confusion matrix for {df_name}: {e}")

                 print(f"     Plotting Calibration Curve...")
                 try:
                    valid_conf_mask = ~y_confidence.isnull()
                    if valid_conf_mask.sum() > 0:
                        prob_true, prob_pred = calibration_curve((y_true == y_pred)[valid_conf_mask], y_confidence[valid_conf_mask], n_bins=10, strategy='uniform')
                        # Apply DPI here
                        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, linestyle='-', label=f'{df_name} Confidence Calibration')
                        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
                        ax.set_xlabel(f"Mean Predicted Confidence ('{proba_col}')")
                        ax.set_ylabel("Fraction of Positives (Accuracy within bin)")
                        ax.set_title(f"'{df_name}' - Confidence Calibration Curve")
                        ax.grid(True); ax.legend(loc="lower right"); plt.tight_layout(); plt.show()
                    else: print(f"     Skipping calibration curve for '{df_name}' (all confidence values are NaN).")
                 except ValueError as ve: print(f"     Warning: Could not generate calibration curve for '{df_name}'. Error: {ve}")
                 except Exception as e: print(f"     Error plotting calibration curve for '{df_name}': {e}")

            processed_datasets.add(df_name)

        except Exception as e:
            print(f"!!! An error occurred processing dataset '{df_name}': {e}")
            import traceback; traceback.print_exc()
            if df_name in results: del results[df_name]


    # --- 2. Plot Combined Performance vs Confidence (Applying DPI) ---
    if plot_charts and processed_datasets:
        print(f"\n2. Plotting Combined Performance vs. Confidence Threshold (DPI={dpi})...")

        # Plot Macro Precision
        plt.figure(figsize=figsize, dpi=dpi) # Apply DPI
        for df_name in processed_datasets:
            if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['macro_precision'], marker='o', linestyle='-', label=f'{df_name} Macro Precision')
        plt.title('Macro Precision (KPI) vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold ({proba_col})')
        plt.ylabel('Macro Precision'); plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout(); plt.show()

        # Plot Accuracy
        plt.figure(figsize=figsize, dpi=dpi) # Apply DPI
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['accuracy'], marker='o', linestyle='-', label=f'{df_name} Accuracy')
        plt.title('Accuracy vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold ({proba_col})')
        plt.ylabel('Accuracy'); plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout(); plt.show()

        # Plot Coverage
        plt.figure(figsize=figsize, dpi=dpi) # Apply DPI
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty:
                conf_df = results[df_name]['by_confidence']
                plt.plot(conf_df['threshold'], conf_df['coverage'], marker='o', linestyle='-', label=f'{df_name} Coverage')
        plt.title('Coverage vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold ({proba_col})')
        plt.ylabel('Coverage (Fraction of Samples)'); plt.grid(True); plt.legend(); plt.ylim(0, 1.05); plt.tight_layout(); plt.show()
    # (Handling for skipping plots remains the same)
    elif not processed_datasets: print("\n2. Skipping combined plots (no datasets processed).")
    else: print("\n2. Skipping combined plots (plot_charts=False).")

    # --- 3. Perform Specified Comparisons ---
    # (Logic remains the same)
    print("\n3. Performing Specified Dataset Comparisons...")
    if comparisons and isinstance(comparisons, list):
        # ... (comparison logic remains the same) ...
        compared_pairs = set()
        for comp_pair in comparisons:
            # ...(validation of comp_pair)...
            if not isinstance(comp_pair, tuple) or len(comp_pair) != 2: print(f"   Skipping invalid comparison: {comp_pair}."); continue
            name1, name2 = comp_pair
            sorted_pair = tuple(sorted((name1, name2)))
            if sorted_pair in compared_pairs: continue
            compared_pairs.add(sorted_pair)
            # ...(check results exist)...
            if name1 not in results or name2 not in results or not results[name1]['overall'] or not results[name2]['overall']: print(f"   Cannot compare '{name1}' vs '{name2}': Results missing."); continue

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
                 val1 = metrics1.get(metric, float('nan')); val2 = metrics2.get(metric, float('nan'))
                 change = val2 - val1 if not (np.isnan(val1) or np.isnan(val2)) else float('nan')
                 print(f"| {metric:<17} | {val1: >{max_name_len}.4f} | {val2: >{max_name_len}.4f} | {change: >+17.4f} |")
                 if metric == kpi_name: kpi_diff = change if not np.isnan(change) else 0.0
            print(f"|-------------------|{'-'*(max_name_len+2)}|{'-'*(max_name_len+2)}|-------------------|")
            # ...(print warning/info)...
            if kpi_diff < -0.03: print(f"   WARNING: Potential performance degradation detected ({kpi_name}... in '{name2}' vs '{name1}').")
            elif kpi_diff > 0.03: print(f"   INFO: Performance ({kpi_name}) is notably higher in '{name2}' vs '{name1}'.")
            else: print(f"   INFO: Performance ({kpi_name}) is similar between '{name1}' and '{name2}'.")
    elif comparisons: print("   'comparisons' argument provided but is not a list.")
    else: print("   No specific comparisons requested.")

    # --- 4. Additional Tests/Suggestions ---
    # (Remains the same)
    print("\n4. Further Analysis Suggestions:")
    print(f"    - Error Analysis: Review where '{pred_col}' != '{target_col}', check '{proba_col}' values.")
    print("    - Check 'by_metadata' results for performance variations in specific groups.")
    # ... other suggestions ...

    print("\n--- Evaluation Complete ---")
    return results

# --- Example Usage (Demonstrating new defaults/args) ---

# (Keep the dummy data generation part as before, ensuring it creates
# 'text', 'RCC_Name', 'predicted_label', 'prediction_confidence',
# 'file_type', 'date_added' columns in df_val, df_test, df_holdout)
# ... (dummy data loading/splitting/prediction generation) ...
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

print("Loading sample data...")
categories = ['sci.med', 'sci.space', 'talk.politics.guns', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=categories)
data = pd.DataFrame({'text': newsgroups.data, 'RCC': newsgroups.target})
data['file_type'] = np.random.choice(['PDF', 'DOCX', 'TXT', 'EMAIL'], size=len(data))
data['date_added'] = pd.to_datetime(pd.Timestamp('2023-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=len(data)), unit='D'))
target_map = {i: name for i, name in enumerate(newsgroups.target_names)}
data['RCC_Name'] = data['RCC'].map(target_map)
df_train, df_temp = train_test_split(data, test_size=0.5, random_state=42, stratify=data['RCC'])
df_val, df_temp2 = train_test_split(df_temp, test_size=0.6, random_state=123, stratify=df_temp['RCC'])
df_test, df_holdout = train_test_split(df_temp2, test_size=0.5, random_state=456, stratify=df_temp2['RCC'])
print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}, Holdout size: {len(df_holdout)}")

print("\nTraining dummy model & generating example predictions...")
temp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('clf', CalibratedClassifierCV(LinearSVC(dual="auto", random_state=42, C=0.1), cv=3))])
temp_pipeline.fit(df_train['text'], df_train['RCC_Name'])
for df_example in [df_val, df_test, df_holdout]:
    df_example['predicted_label'] = temp_pipeline.predict(df_example['text'])
    probas = temp_pipeline.predict_proba(df_example['text'])
    df_example['prediction_confidence'] = np.max(probas, axis=1)
    noise = np.random.uniform(-0.05, 0.05, size=len(df_example))
    df_example['prediction_confidence'] = np.clip(df_example['prediction_confidence'] + noise, 0, 1)
print("Example prediction columns added.")


# --- Run the evaluation function with new settings ---
datasets_to_evaluate = {
    'Validation': df_val,
    'Test': df_test,
    'Holdout_2024': df_holdout
}
metadata_cols_to_analyze = ['file_type', 'date_added']
comparisons_to_make = [ ('Validation', 'Test'), ('Test', 'Holdout_2024') ]

evaluation_results = evaluate_precomputed_predictions(
    datasets=datasets_to_evaluate,
    text_col='text',
    target_col='RCC_Name',
    pred_col='predicted_label',
    proba_col='prediction_confidence',
    metadata_cols=metadata_cols_to_analyze,
    comparisons=comparisons_to_make,
    # confidence_thresholds=np.arange(0.05, 1.05, 0.05), # Now default
    plot_charts=True,
    dpi=150 # Example of setting higher DPI
)

# Now you should see metadata summary tables printed during execution
# And the plots should have a higher resolution (150 DPI)
