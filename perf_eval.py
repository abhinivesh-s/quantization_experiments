# Annotate each point with its y-value
for i in range(len(x)):
    ax1.text(x[i], y[i], f'{y[i]:.2f}', ha='center', va='bottom')




# Find texts associated with multiple classes
conflicting_texts = df.groupby('text')['class'].nunique()
conflicting_texts = conflicting_texts[conflicting_texts > 1].index

# Drop rows where text is in the list of conflicting texts
df_cleaned = df[~df['text'].isin(conflicting_texts)]



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
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype

# Suppress UndefinedMetricWarning for cases where a class/group might be missing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential division by zero with empty groups

# --- Helper Formatting Functions ---
def format_metric(value, precision=4):
    """Helper to format metrics, handling NaNs."""
    if pd.isna(value):
        return "-"
    return f"{value:.{precision}f}"

def format_percentage(value, precision=1):
    """Helper to format coverage percentage, handling NaNs."""
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{precision}f}%"

def format_count(value):
    """Helper to format counts, handling NaNs."""
    if pd.isna(value):
        return "-"
    # Check if value can be converted to int (might be float NaN)
    try:
        return f"{int(value):d}"
    except (ValueError, TypeError):
        return "-"
# --- End Helper Functions ---

def evaluate_precomputed_predictions(
    datasets: Dict[str, pd.DataFrame],
    text_col: str, # Optional context
    target_col: str,
    pred_col: str,
    proba_col: str,
    metadata_cols: Optional[List[str]] = None,
    datetime_metadata_col: Optional[str] = None, # Separate datetime column arg
    comparisons: Optional[List[Tuple[str, str]]] = None,
    confidence_thresholds: np.ndarray = np.arange(0.05, 1.05, 0.05),
    plot_charts: bool = True,
    figsize: tuple = (12, 6),
    dpi: int = 100,
    cm_figsize_scale: float = 0.5,
    max_cm_size: int = 25,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    year_bucket_size: int = 1
):
    """
    Performs comprehensive performance testing using pre-computed predictions
    and probabilities. Includes Markdown tables for metadata/confidence summaries,
    separate datetime handling, and DPI setting.

    (Args description remains the same)

    Returns:
        dict: Dictionary containing performance results keyed by dataset name.
    """
    print("--- Starting Performance Evaluation from Pre-computed Predictions ---")
    if not datasets: print("No datasets provided. Exiting."); return {}

    if not isinstance(year_bucket_size, int) or year_bucket_size < 1:
        print(f"Warning: Invalid year_bucket_size ({year_bucket_size}). Defaulting to 1.")
        year_bucket_size = 1

    all_metadata_cols_to_check = []
    if metadata_cols: all_metadata_cols_to_check.extend(metadata_cols)
    if datetime_metadata_col: all_metadata_cols_to_check.append(datetime_metadata_col)
    all_metadata_cols_to_check = list(set(all_metadata_cols_to_check))

    results = {}
    all_class_names_ref = None
    # ----------------------------------------------------------
    # <<< CORRECT PLACEMENT FOR processed_datasets INITIALIZATION >>>
    processed_datasets = set()
    # ----------------------------------------------------------

    # --- Function to process a single metadata column (remains the same) ---
    def process_metadata_column(df_eval, col_name, is_datetime=False):
        # (Function implementation as in the previous correct response)
        # Uses `results`, `df_name`, `current_class_names` from outer scope
        print(f"     ----- Analyzing by Metadata Column: '{col_name}' -----")
        metadata_results_list = []
        # Need to access results dict, df_name etc from the outer scope where this is called
        if df_name not in results: # Safety check if called unexpectedly
             print(f"Error: Cannot process metadata for {df_name} as results dict not initialized.")
             return
        if col_name not in results[df_name]['by_metadata']:
             results[df_name]['by_metadata'][col_name] = {}

        binned_col_name = f"{col_name}_binned"
        grouped = None
        # Datetime Logic
        if is_datetime:
            # ... (datetime binning logic as before) ...
             print(f"       Applying year-based binning (size={year_bucket_size}).")
             binned_col_name = f"{col_name}_year_bucket"
             try:
                if not is_datetime64_any_dtype(df_eval[col_name]): df_eval[col_name] = pd.to_datetime(df_eval[col_name], errors='coerce')
                if df_eval[col_name].isnull().all(): raise ValueError("All datetime values NaT.")
                years = df_eval[col_name].dt.year
                current_year_start = year_start if year_start is not None else int(years.min())
                current_year_end = year_end if year_end is not None else int(years.max())
                if year_start is None: print(f"       Auto-detected start year: {current_year_start}")
                if year_end is None: print(f"       Auto-detected end year: {current_year_end}")
                if np.isnan(current_year_start) or np.isnan(current_year_end): raise ValueError("Invalid start/end year.")
                bins = np.arange(current_year_start, current_year_end + 1 + 0.1, year_bucket_size)
                if len(bins) < 2:
                    print(f"       Warning: Not enough year range. Single group.")
                    df_eval[binned_col_name] = f"{int(current_year_start)}-{int(current_year_end)}"
                    grouped = df_eval.groupby(binned_col_name, observed=False)
                else:
                    labels = [f"{int(bins[i])}" if year_bucket_size == 1 else f"{int(bins[i])}-{int(bins[i+1] - 1)}" for i in range(len(bins) - 1)]
                    df_eval[binned_col_name] = pd.cut(years, bins=bins, labels=labels, right=False, include_lowest=True)
                    grouped = df_eval.groupby(binned_col_name, observed=False)
                    print(f"       Created year bins: {labels}")
             except Exception as e:
                 print(f"       Error during year-binning for '{col_name}': {e}. Skipping."); results[df_name]['by_metadata'][col_name] = {"ERROR": f"Year binning failed: {e}"}; return
        # Categorical Logic
        elif is_categorical_dtype(df_eval[col_name]) or (df_eval[col_name].dtype == 'object' and df_eval[col_name].nunique() < 20):
             # ... (categorical grouping logic as before) ...
             print(f"       Treating '{col_name}' as categorical.")
             binned_col_name = col_name
             if df_eval[col_name].dtype == 'object':
                try: df_eval[col_name] = df_eval[col_name].astype('category')
                except Exception: print(f"       Warning: Could not convert '{col_name}' to category.")
             grouped = df_eval.groupby(binned_col_name, observed=False)
        # Other Continuous Logic
        else:
             # ... (continuous binning logic as before) ...
             print(f"       Treating '{col_name}' as continuous, general binning.")
             try:
                try: df_eval[binned_col_name] = pd.qcut(df_eval[col_name].astype(float), q=5, duplicates='drop'); print("       (Binned into 5 quantiles)")
                except (ValueError, TypeError): df_eval[binned_col_name] = pd.cut(df_eval[col_name].astype(float), bins=5); print("       (Binned into 5 equal-width intervals)")
                grouped = df_eval.groupby(binned_col_name, observed=False)
             except Exception as e:
                print(f"       Warning: Could not bin '{col_name}'. Skipping. Error: {e}"); results[df_name]['by_metadata'][col_name] = {"ERROR": f"Binning failed: {e}"}; return

        if grouped is None: print(f"       Error: Grouping strategy failed for '{col_name}'. Skipping."); results[df_name]['by_metadata'][col_name] = {"ERROR": "Grouping failed"}; return

        # Calculate Metrics per Group
        for grp_name, group in grouped:
            # ... (metric calculation logic as before) ...
            grp_name_str = "NaN/Missing" if pd.isna(grp_name) else str(grp_name)
            group_metrics = {'num_samples': len(group)}
            if len(group) > 0:
                 grp_y_true = group[target_col]; grp_y_pred = group[pred_col]
                 group_metrics['accuracy'] = accuracy_score(grp_y_true, grp_y_pred)
                 if len(group) >= 2 and grp_y_true.nunique() >= 2:
                     try:
                         # Use current_class_names from outer scope
                         group_metrics['macro_precision'] = precision_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                         group_metrics['macro_recall'] = recall_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                         group_metrics['macro_f1'] = f1_score(grp_y_true, grp_y_pred, average='macro', zero_division=0, labels=current_class_names)
                     except Exception: group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan
                 else: group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan
            else: group_metrics['accuracy'], group_metrics['macro_precision'], group_metrics['macro_recall'], group_metrics['macro_f1'] = np.nan, np.nan, np.nan, np.nan
            results[df_name]['by_metadata'][col_name][grp_name_str] = group_metrics
            group_metrics['group_name'] = grp_name_str
            metadata_results_list.append(group_metrics)

        # Print Metadata Summary Table (Markdown)
        if metadata_results_list:
            # ... (Markdown table printing logic as before) ...
            meta_df = pd.DataFrame(metadata_results_list).sort_values(by='group_name')
            max_grp_len = meta_df['group_name'].astype(str).map(len).max()
            max_grp_len = max(max_grp_len if not pd.isna(max_grp_len) else 0, len(col_name) + 6, 15)
            print(f"\n       **Metadata Summary: {col_name}**")
            print(f"| {'Group':<{max_grp_len}} | Samples  | Accuracy | Macro Precision |")
            print(f"| :{'-'*(max_grp_len-1)} | :------: | :------: | :-------------: |")
            for _, row in meta_df.iterrows():
                print(f"| {str(row['group_name']):<{max_grp_len}} | {format_count(row['num_samples']):>8} | {format_metric(row.get('accuracy')):>8} | {format_metric(row.get('macro_precision')):>15} |")
            print("\n")
        else: print(f"       No groups processed for '{col_name}'.")
        print(f"     ----- Finished Analyzing: '{col_name}' -----")
    # --- End of process_metadata_column function ---


    # --- 1. Process each dataset ---
    for df_name, df in datasets.items():
        print(f"\n--- Processing Dataset: {df_name} ---")
        # ... basic dataset validation ...
        if df is None or df.empty: print(f"Skipping '{df_name}' (empty/None)."); continue
        if not isinstance(df, pd.DataFrame): print(f"Skipping '{df_name}' (not DataFrame)."); continue
        required_cols = [target_col, pred_col, proba_col]; missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: print(f"Skipping '{df_name}': Missing cols: {missing_cols}."); continue
        missing_meta = [m_col for m_col in all_metadata_cols_to_check if m_col not in df.columns]
        if missing_meta: print(f"   Warning: Metadata columns not found in '{df_name}': {missing_meta}")

        # Initialize results structure for this dataset
        results[df_name] = {
            'overall': {}, 'by_metadata': {}, 'by_confidence': pd.DataFrame(),
            'class_names': None, 'y_true': None, 'y_pred': None, 'y_confidence': None
        }

        try:
            print(f"   Dataset '{df_name}' ({len(df)} samples)")
            df_eval = df.copy()

            # --- 1a. Get Labels, Predictions, Confidence ---
            # ... (logic as before) ...
            print("   Reading pre-computed data...")
            y_true = df_eval[target_col]; y_pred = df_eval[pred_col]; y_confidence = df_eval[proba_col]
            if not (y_confidence.min() >= 0 and y_confidence.max() <= 1): print(f"   Warning: Probabilities outside [0, 1].")
            if y_confidence.isnull().any(): print(f"   Warning: NaNs in probabilities.")
            results[df_name]['y_true'] = y_true; results[df_name]['y_pred'] = y_pred; results[df_name]['y_confidence'] = y_confidence
            current_class_names = sorted(y_true.astype(str).unique())
            results[df_name]['class_names'] = current_class_names
            if all_class_names_ref is None: all_class_names_ref = current_class_names; print(f"   Reference classes: {all_class_names_ref}")
            elif not np.array_equal(all_class_names_ref, current_class_names): print(f"   Warning: True classes differ.")


            # --- 1b. Overall Performance ---
            # ... (logic as before) ...
            print("   Calculating Overall Performance...")
            metrics = results[df_name]['overall']; metrics['num_samples'] = len(y_true)
            try:
                 metrics['accuracy'] = accuracy_score(y_true, y_pred)
                 metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                 metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                 metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=current_class_names)
                 metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                 metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                 metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=current_class_names)
                 print(f"     Accuracy:         {format_metric(metrics['accuracy'])}")
                 print(f"     Macro Precision:  {format_metric(metrics['macro_precision'])} (KPI)")
                 print(f"     Macro Recall:     {format_metric(metrics['macro_recall'])}")
                 print(f"     Macro F1:         {format_metric(metrics['macro_f1'])}")
            except Exception as e:
                 print(f"     Error calculating overall metrics: {e}")
                 for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1']: metrics[k] = np.nan


            # --- 1c. Performance by Metadata Columns ---
            print("\n   Calculating Performance by Metadata Columns...")
            # Process datetime column first if specified
            if datetime_metadata_col and datetime_metadata_col in df_eval.columns:
                process_metadata_column(df_eval, datetime_metadata_col, is_datetime=True)
            elif datetime_metadata_col:
                 print(f"   Skipping specified datetime column '{datetime_metadata_col}' (not found).")

            # Process other metadata columns
            if metadata_cols:
                for col in metadata_cols:
                    if col == datetime_metadata_col: continue # Avoid double processing
                    if col not in df_eval.columns:
                        print(f"     Warning: Metadata column '{col}' not found. Skipping.")
                        continue
                    process_metadata_column(df_eval, col, is_datetime=False) # is_datetime=False for these


            # --- 1d. Performance by Confidence Threshold ---
            # ... (logic as before to calculate conf_df) ...
            print("\n   Calculating Performance by Confidence Threshold...")
            confidence_results_list = []
            total_samples = len(y_true); y_confidence_filled = y_confidence.fillna(-1)
            for threshold in confidence_thresholds:
                mask = y_confidence_filled >= threshold
                covered_samples = np.sum(mask)
                coverage = covered_samples / total_samples if total_samples > 0 else 0
                thresh_metrics = {'threshold': threshold, 'coverage': coverage, 'num_samples_covered': covered_samples}
                if covered_samples > 0:
                    y_true_thresh = y_true[mask]; y_pred_thresh = y_pred[mask]
                    thresh_metrics['accuracy'] = accuracy_score(y_true_thresh, y_pred_thresh)
                    if y_true_thresh.nunique() > 1:
                        try:
                            thresh_metrics['macro_precision'] = precision_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                            thresh_metrics['macro_recall'] = recall_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                            thresh_metrics['macro_f1'] = f1_score(y_true_thresh, y_pred_thresh, average='macro', zero_division=0, labels=current_class_names)
                        except Exception: thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan
                    else: thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan
                else: thresh_metrics['accuracy'], thresh_metrics['macro_precision'], thresh_metrics['macro_recall'], thresh_metrics['macro_f1'] = np.nan, np.nan, np.nan, np.nan
                confidence_results_list.append(thresh_metrics)
            conf_df = pd.DataFrame(confidence_results_list)
            results[df_name]['by_confidence'] = conf_df
            print(f"     Analyzed {len(confidence_thresholds)} confidence thresholds.")


            # --- 1e. Confidence Cutoff Summary Table ---
            # ... (Markdown table printing using conf_df and format helpers)...
            print(f"\n   **Confidence Threshold Summary: {df_name}**")
            print(f"| Threshold | Accuracy | Macro Precision | Coverage | Samples Covered |")
            print(f"| :-------- | :------: | :-------------: | :------: | :-------------: |")
            for _, row in conf_df.iterrows():
                print(f"| {format_metric(row['threshold'], 2):<9} | {format_metric(row.get('accuracy')):>8} | {format_metric(row.get('macro_precision')):>15} | {format_percentage(row.get('coverage')):>8} | {format_count(row.get('num_samples_covered')):>15} |")
            print("\n")


            # --- 1f. Detailed Classification Report ---
            # ... (logic as before) ...
            print("\n   Generating Classification Report...")
            try:
                report = classification_report(y_true, y_pred, target_names=current_class_names, zero_division=0, labels=current_class_names)
                print(report)
                results[df_name]['overall']['classification_report'] = report
            except Exception as e: print(f"     Error generating classification report: {e}"); results[df_name]['overall']['classification_report'] = "Error."


            # --- 1g. Plotting ---
            # ... (logic as before) ...
            if plot_charts:
                 print(f"\n   Plotting Charts for '{df_name}' (DPI={dpi})...")
                 # Confusion Matrix
                 try:
                    num_classes = len(current_class_names); cm_size = min(max(6, num_classes * cm_figsize_scale), max_cm_size)
                    fig, ax = plt.subplots(figsize=(cm_size, cm_size), dpi=dpi)
                    cm = confusion_matrix(y_true, y_pred, labels=current_class_names)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_class_names)
                    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical'); plt.title(f"'{df_name}' - Confusion Matrix"); plt.tight_layout(); plt.show()
                 except Exception as e: print(f"     Error plotting confusion matrix: {e}")
                 # Calibration Curve
                 try:
                     valid_conf_mask = ~y_confidence.isnull()
                     if valid_conf_mask.sum() > 0:
                         prob_true, prob_pred = calibration_curve((y_true == y_pred)[valid_conf_mask], y_confidence[valid_conf_mask], n_bins=10, strategy='uniform')
                         fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                         ax.plot(prob_pred, prob_true, marker='o', linewidth=1, linestyle='-', label=f'{df_name} Confidence Calibration')
                         ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
                         ax.set_xlabel(f"Mean Predicted Confidence ('{proba_col}')"); ax.set_ylabel("Fraction of Positives (Accuracy within bin)")
                         ax.set_title(f"'{df_name}' - Confidence Calibration Curve"); ax.grid(True); ax.legend(loc="lower right"); plt.tight_layout(); plt.show()
                     else: print(f"     Skipping calibration curve ('{df_name}', all confidence NaN).")
                 except ValueError as ve: print(f"     Warning: Could not generate calibration curve ('{df_name}'). Error: {ve}")
                 except Exception as e: print(f"     Error plotting calibration curve ('{df_name}'): {e}")


            # --- Mark as processed successfully ---
            processed_datasets.add(df_name) # <<< THIS IS THE CRITICAL LINE

        except Exception as e:
            print(f"!!! An error occurred processing dataset '{df_name}': {e}")
            import traceback; traceback.print_exc()
            # Ensure partial results are cleaned up if error occurs mid-processing
            if df_name in results: del results[df_name]
            # Ensure it's not marked as processed if error occurs
            if df_name in processed_datasets: processed_datasets.remove(df_name)


    # --- 2. Plot Combined Performance vs Confidence ---
    # Check if processed_datasets is not empty before proceeding
    if plot_charts and processed_datasets:
        print(f"\n2. Plotting Combined Performance vs. Confidence Threshold (DPI={dpi})...")
        # ... (Combined plots using dpi=dpi and iterating over processed_datasets) ...
        # Macro Precision
        plt.figure(figsize=figsize, dpi=dpi)
        for df_name in processed_datasets: # Iterate over the set of successfully processed names
            if df_name in results and not results[df_name]['by_confidence'].empty: plt.plot(results[df_name]['by_confidence']['threshold'], results[df_name]['by_confidence']['macro_precision'], marker='o', linestyle='-', label=f'{df_name} Macro Precision')
        plt.title('Macro Precision (KPI) vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold ({proba_col})'); plt.ylabel('Macro Precision'); plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout(); plt.show()
        # Accuracy
        plt.figure(figsize=figsize, dpi=dpi)
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty: plt.plot(results[df_name]['by_confidence']['threshold'], results[df_name]['by_confidence']['accuracy'], marker='o', linestyle='-', label=f'{df_name} Accuracy')
        plt.title('Accuracy vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold ({proba_col})'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout(); plt.show()
        # Coverage
        plt.figure(figsize=figsize, dpi=dpi)
        for df_name in processed_datasets:
             if df_name in results and not results[df_name]['by_confidence'].empty: plt.plot(results[df_name]['by_confidence']['threshold'], results[df_name]['by_confidence']['coverage'], marker='o', linestyle='-', label=f'{df_name} Coverage')
        plt.title('Coverage vs. Confidence Threshold'); plt.xlabel(f'Confidence Threshold ({proba_col})'); plt.ylabel('Coverage (% Samples)'); plt.grid(True); plt.legend(); plt.ylim(0, 1.05); plt.tight_layout(); plt.show()

    elif not processed_datasets: # Check if the set is empty
         print("\n2. Skipping combined plots (no datasets processed successfully).")
    else: # plot_charts is False
        print("\n2. Skipping combined plots (plot_charts=False).")


    # --- 3. Perform Specified Comparisons (Markdown Table) ---
    # (Logic using Markdown remains the same)
    print("\n3. Performing Specified Dataset Comparisons...")
    if comparisons and isinstance(comparisons, list):
        # ... (comparison logic using Markdown as before) ...
         compared_pairs = set()
         for comp_pair in comparisons:
            if not isinstance(comp_pair, tuple) or len(comp_pair) != 2: print(f"   Skipping invalid comparison: {comp_pair}."); continue
            name1, name2 = comp_pair; sorted_pair = tuple(sorted((name1, name2)))
            if sorted_pair in compared_pairs: continue; compared_pairs.add(sorted_pair)
            if name1 not in results or name2 not in results or not results[name1]['overall'] or not results[name2]['overall']: print(f"   Cannot compare '{name1}' vs '{name2}': Results missing."); continue
            print(f"\n   **Comparison: {name1} vs {name2}**"); metrics1 = results[name1]['overall']; metrics2 = results[name2]['overall']
            max_name_len = max(len(name1), len(name2), 7); header1 = f"{name1:<{max_name_len}}"; header2 = f"{name2:<{max_name_len}}"
            print(f"| Metric            | {header1} | {header2} | Change ({name2}-{name1}) |")
            print(f"| :---------------- | :{'-'*(max_name_len-1)} | :{'-'*(max_name_len-1)} | :----------------: |")
            kpi_diff = 0.0; kpi_name = 'macro_precision'
            for metric in ['accuracy', kpi_name, 'macro_recall', 'macro_f1']:
                 val1 = metrics1.get(metric, float('nan')); val2 = metrics2.get(metric, float('nan')); change = val2 - val1 if not (np.isnan(val1) or np.isnan(val2)) else float('nan')
                 print(f"| {metric:<17} | {format_metric(val1):>{max_name_len}} | {format_metric(val2):>{max_name_len}} | {format_metric(change, 4):>18} |")
                 if metric == kpi_name: kpi_diff = change if not np.isnan(change) else 0.0
            print("\n")
            if kpi_diff < -0.03: print(f"   *WARNING:* Potential performance degradation detected ({kpi_name}... in '{name2}' vs '{name1}').")
            elif kpi_diff > 0.03: print(f"   *INFO:* Performance ({kpi_name}) notably higher in '{name2}' vs '{name1}'.")
            else: print(f"   *INFO:* Performance ({kpi_name}) similar between '{name1}' and '{name2}'.")

    elif comparisons: print("   'comparisons' argument provided but is not a list.")
    else: print("   No specific comparisons requested.")


    # --- 4. Additional Tests/Suggestions --- (No changes)
    print("\n4. Further Analysis Suggestions:")
    # ... (Suggestions) ...

    print("\n--- Evaluation Complete ---")
    return results


# --- Example Usage (remains the same) ---
# ... (Load data, train temp model, generate predictions as before) ...
# ... (Define datasets_to_evaluate, datetime_col, other_metadata, comparisons_to_make) ...

# evaluation_results = evaluate_precomputed_predictions(
#     datasets=datasets_to_evaluate,
#     text_col='text',
#     target_col='RCC_Name',
#     pred_col='predicted_label',
#     proba_col='prediction_confidence',
#     metadata_cols=other_metadata,
#     datetime_metadata_col=datetime_col,
#     comparisons=comparisons_to_make,
#     plot_charts=True,
#     dpi=100,
#     year_start=None,
#     year_end=None,
#     year_bucket_size=1
# )
