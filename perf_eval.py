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
# No seaborn needed if not used explicitly for styling beyond matplotlib defaults
# import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype

# Suppress UndefinedMetricWarning for cases where a class/group might be missing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For potential division by zero with empty groups

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
    return f"{int(value):d}"

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

    Args:
        datasets: Dict of dataset names to DataFrames.
        text_col: Name of the original text column (optional context).
        target_col: Name of the true label column.
        pred_col: Name of the predicted label column.
        proba_col: Name of the confidence score/probability column.
        metadata_cols: List of *non-datetime* metadata columns for general analysis.
        datetime_metadata_col: Specific datetime column for year-based bucketing.
        comparisons: Dataset pairs to compare.
        confidence_thresholds: Confidence thresholds to evaluate.
        plot_charts: Generate plots.
        figsize: Default figure size.
        dpi: Plot resolution.
        cm_figsize_scale: Confusion matrix size scaling.
        max_cm_size: Max confusion matrix dimension.
        year_start: Start year for bucketing the datetime_metadata_col. (None=auto)
        year_end: End year for bucketing the datetime_metadata_col. (None=auto)
        year_bucket_size: Size (in years) of buckets for datetime_metadata_col.

    Returns:
        dict: Dictionary containing performance results keyed by dataset name.
    """
    print("--- Starting Performance Evaluation from Pre-computed Predictions ---")
    if not datasets: print("No datasets provided. Exiting."); return {}

    if not isinstance(year_bucket_size, int) or year_bucket_size < 1:
        print(f"Warning: Invalid year_bucket_size ({year_bucket_size}). Defaulting to 1.")
        year_bucket_size = 1

    # Combine metadata columns for initial check, ensuring no duplicates
    all_metadata_cols_to_check = []
    if metadata_cols:
        all_metadata_cols_to_check.extend(metadata_cols)
    if datetime_metadata_col:
        all_metadata_cols_to_check.append(datetime_metadata_col)
    all_metadata_cols_to_check = list(set(all_metadata_cols_to_check)) # Unique list

    results = {}
    all_class_names_ref = None

    # --- Function to process a single metadata column (refactored for reuse) ---
    def process_metadata_column(df_eval, col_name, is_datetime=False):
        print(f"     ----- Analyzing by Metadata Column: '{col_name}' -----")
        metadata_results_list = []
        results[df_name]['by_metadata'][col_name] = {}
        binned_col_name = f"{col_name}_binned"

        # Grouping Logic
        grouped = None
        if is_datetime:
            print(f"       Applying year-based binning (size={year_bucket_size}).")
            binned_col_name = f"{col_name}_year_bucket"
            try:
                # Convert if not already datetime, handling errors
                if not is_datetime64_any_dtype(df_eval[col_name]):
                     df_eval[col_name] = pd.to_datetime(df_eval[col_name], errors='coerce')
                if df_eval[col_name].isnull().all(): raise ValueError("All datetime values are NaT.")

                years = df_eval[col_name].dt.year
                current_year_start = year_start if year_start is not None else int(years.min())
                current_year_end = year_end if year_end is not None else int(years.max())
                if year_start is None: print(f"       Auto-detected start year: {current_year_start}")
                if year_end is None: print(f"       Auto-detected end year: {current_year_end}")
                if np.isnan(current_year_start) or np.isnan(current_year_end): raise ValueError("Could not determine valid start/end year.")

                bins = np.arange(current_year_start, current_year_end + 1 + 0.1, year_bucket_size)
                if len(bins) < 2:
                    print(f"       Warning: Not enough year range. Treating as single group.")
                    df_eval[binned_col_name] = f"{int(current_year_start)}-{int(current_year_end)}"
                    grouped = df_eval.groupby(binned_col_name, observed=False)
                else:
                    labels = [f"{int(bins[i])}" if year_bucket_size == 1 else f"{int(bins[i])}-{int(bins[i+1] - 1)}" for i in range(len(bins) - 1)]
                    df_eval[binned_col_name] = pd.cut(years, bins=bins, labels=labels, right=False, include_lowest=True)
                    grouped = df_eval.groupby(binned_col_name, observed=False)
                    print(f"       Created year bins: {labels}")

            except Exception as e:
                 print(f"       Error during year-based binning for '{col_name}': {e}. Skipping.")
                 results[df_name]['by_metadata'][col_name] = {"ERROR": f"Year binning failed: {e}"}
                 return # Exit this column's processing
        # Categorical Logic
        elif is_categorical_dtype(df_eval[col_name]) or (df_eval[col_name].dtype == 'object' and df_eval[col_name].nunique() < 20):
            print(f"       Treating '{col_name}' as categorical.")
            binned_col_name = col_name
            if df_eval[col_name].dtype == 'object':
                try: df_eval[col_name] = df_eval[col_name].astype('category')
                except Exception: print(f"       Warning: Could not convert '{col_name}' to category.")
            grouped = df_eval.groupby(binned_col_name, observed=False)
        # Other Continuous Logic
        else:
            print(f"       Treating '{col_name}' as continuous, attempting general binning.")
            try:
                # Prefer qcut, fallback to cut
                try: df_eval[binned_col_name] = pd.qcut(df_eval[col_name].astype(float), q=5, duplicates='drop'); print("       (Binned into 5 quantiles)")
                except (ValueError, TypeError): df_eval[binned_col_name] = pd.cut(df_eval[col_name].astype(float), bins=5); print("       (Binned into 5 equal-width intervals)")
                grouped = df_eval.groupby(binned_col_name, observed=False)
            except Exception as e:
                print(f"       Warning: Could not bin continuous column '{col_name}'. Skipping. Error: {e}")
                results[df_name]['by_metadata'][col_name] = {"ERROR": f"Binning failed: {e}"}
                return # Exit this column's processing

        if grouped is None: # Should not happen if logic is correct, but safety check
             print(f"       Error: Could not determine grouping strategy for '{col_name}'. Skipping.")
             results[df_name]['by_metadata'][col_name] = {"ERROR": "Grouping strategy failed"}
             return

        # Calculate Metrics per Group
        for grp_name, group in grouped:
            grp_name_str = "NaN/Missing" if pd.isna(grp_name) else str(grp_name)
            group_metrics = {'num_samples': len(group)}
            if len(group) > 0:
                 grp_y_true = group[target_col]; grp_y_pred = group[pred_col]
                 group_metrics['accuracy'] = accuracy_score(grp_y_true, grp_y_pred)
                 if len(group) >= 2 and grp_y_true.nunique() >= 2:
                     try:
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
            meta_df = pd.DataFrame(metadata_results_list).sort_values(by='group_name')
            # Find max length for alignment, ensure minimum width
            max_grp_len = meta_df['group_name'].astype(str).map(len).max()
            max_grp_len = max(max_grp_len if not pd.isna(max_grp_len) else 0, len(col_name) + 6, 15) # Ensure header fits, min width 15

            print(f"\n       **Metadata Summary: {col_name}**")
            # Header row
            print(f"| {'Group':<{max_grp_len}} | Samples  | Accuracy | Macro Precision |")
            # Separator row with alignment hints (:--, :----:, etc.)
            print(f"| :{'-'*(max_grp_len-1)} | :------: | :------: | :-------------: |")
            # Data rows
            for _, row in meta_df.iterrows():
                print(f"| {str(row['group_name']):<{max_grp_len}} | {format_count(row['num_samples']):>8} | {format_metric(row.get('accuracy')):>8} | {format_metric(row.get('macro_precision')):>15} |")
            print("\n") # Add a blank line after table
        else:
            print(f"       No groups processed for '{col_name}'.")
        print(f"     ----- Finished Analyzing: '{col_name}' -----")
    # --- End of process_metadata_column function ---


    # --- 1. Process each dataset ---
    for df_name, df in datasets.items():
        print(f"\n--- Processing Dataset: {df_name} ---")
        # ... (basic dataset validation) ...
        if df is None or df.empty: print(f"Skipping '{df_name}' (empty/None)."); continue
        # ... check required columns ...
        required_cols = [target_col, pred_col, proba_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: print(f"Skipping '{df_name}': Missing cols: {missing_cols}."); continue
        # ... check all specified metadata columns exist ...
        missing_meta = [m_col for m_col in all_metadata_cols_to_check if m_col not in df.columns]
        if missing_meta: print(f"   Warning: Metadata columns not found in '{df_name}': {missing_meta}")


        results[df_name] = {
            'overall': {}, 'by_metadata': {}, 'by_confidence': pd.DataFrame(),
            'class_names': None, 'y_true': None, 'y_pred': None, 'y_confidence': None
        }

        try:
            print(f"   Dataset '{df_name}' ({len(df)} samples)")
            df_eval = df.copy()

            # --- 1a. Get Labels, Predictions, Confidence --- (No changes)
            print("   Reading pre-computed data...")
            # ... (y_true, y_pred, y_confidence assignment and checks) ...
            y_true = df_eval[target_col]; y_pred = df_eval[pred_col]; y_confidence = df_eval[proba_col]
            # ... (confidence validation, class name derivation) ...
            if not (y_confidence.min() >= 0 and y_confidence.max() <= 1): print(f"   Warning: Probabilities in '{proba_col}' outside [0, 1].")
            if y_confidence.isnull().any(): print(f"   Warning: NaNs found in '{proba_col}'.")
            results[df_name]['y_true'] = y_true; results[df_name]['y_pred'] = y_pred; results[df_name]['y_confidence'] = y_confidence
            current_class_names = sorted(y_true.astype(str).unique())
            results[df_name]['class_names'] = current_class_names
            if all_class_names_ref is None: all_class_names_ref = current_class_names; print(f"   Reference classes: {all_class_names_ref}")
            elif not np.array_equal(all_class_names_ref, current_class_names): print(f"   Warning: True classes differ: {current_class_names} vs {all_class_names_ref}")

            # --- 1b. Overall Performance --- (No changes)
            print("   Calculating Overall Performance...")
            # ... (overall metric calculations and printing) ...
            metrics = results[df_name]['overall']; metrics['num_samples'] = len(y_true)
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
            except Exception as e: print(f"     Error calculating overall metrics: {e}"); # Assign NaNs...

            # --- 1c. Performance by Metadata Columns --- (MODIFIED)
            print("\n   Calculating Performance by Metadata Columns...")
            # Process datetime column first if specified
            if datetime_metadata_col and datetime_metadata_col in df_eval.columns:
                process_metadata_column(df_eval, datetime_metadata_col, is_datetime=True)
            else:
                 if datetime_metadata_col: print(f"   Skipping specified datetime column '{datetime_metadata_col}' (not found).")


            # Process other metadata columns
            if metadata_cols:
                for col in metadata_cols:
                    if col == datetime_metadata_col: continue # Avoid reprocessing if passed in both args
                    if col not in df_eval.columns:
                        print(f"     Warning: Metadata column '{col}' not found. Skipping.")
                        continue
                    # Call the processing function (automatically detects type now)
                    process_metadata_column(df_eval, col, is_datetime=False) # Explicitly False here


            # --- 1d. Performance by Confidence Threshold --- (Calculation unchanged)
            print("\n   Calculating Performance by Confidence Threshold...")
            # ... (confidence threshold analysis logic) ...
            confidence_results_list = []
            total_samples = len(y_true)
            y_confidence_filled = y_confidence.fillna(-1)
            for threshold in confidence_thresholds:
                mask = y_confidence_filled >= threshold
                # ... (calculate coverage, accuracy, precision etc. for threshold) ...
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


            # --- 1e. Confidence Cutoff Summary Table (NEW) ---
            print(f"\n   **Confidence Threshold Summary: {df_name}**")
            print(f"| Threshold | Accuracy | Macro Precision | Coverage | Samples Covered |")
            print(f"| :-------- | :------: | :-------------: | :------: | :-------------: |")
            for _, row in conf_df.iterrows():
                print(f"| {format_metric(row['threshold'], 2):<9} | {format_metric(row.get('accuracy')):>8} | {format_metric(row.get('macro_precision')):>15} | {format_percentage(row.get('coverage')):>8} | {format_count(row.get('num_samples_covered')):>15} |")
            print("\n")


            # --- 1f. Detailed Classification Report --- (No changes)
            print("\n   Generating Classification Report...")
            # ... (classification report generation and printing) ...
            try:
                report = classification_report(y_true, y_pred, target_names=current_class_names, zero_division=0, labels=current_class_names)
                print(report)
                results[df_name]['overall']['classification_report'] = report
            except Exception as e: print(f"     Error generating classification report: {e}"); results[df_name]['overall']['classification_report'] = "Error."


            # --- 1g. Plotting --- (No changes)
            if plot_charts:
                print(f"\n   Plotting Charts for '{df_name}' (DPI={dpi})...")
                # ... (Confusion Matrix plotting with dpi=dpi) ...
                # ... (Calibration Curve plotting with dpi=dpi) ...


            processed_datasets.add(df_name)

        except Exception as e:
            print(f"!!! An error occurred processing dataset '{df_name}': {e}")
            import traceback; traceback.print_exc()
            if df_name in results: del results[df_name]


    # --- 2. Plot Combined Performance vs Confidence --- (No changes)
    if plot_charts and processed_datasets:
        print(f"\n2. Plotting Combined Performance vs. Confidence Threshold (DPI={dpi})...")
        # ... (Combined plots with dpi=dpi) ...


    # --- 3. Perform Specified Comparisons (Markdown Table) ---
    print("\n3. Performing Specified Dataset Comparisons...")
    if comparisons and isinstance(comparisons, list):
        compared_pairs = set()
        for comp_pair in comparisons:
            # ... (validation of comp_pair) ...
            if not isinstance(comp_pair, tuple) or len(comp_pair) != 2: print(f"   Skipping invalid comparison: {comp_pair}."); continue
            name1, name2 = comp_pair; sorted_pair = tuple(sorted((name1, name2)))
            if sorted_pair in compared_pairs: continue; compared_pairs.add(sorted_pair)
            if name1 not in results or name2 not in results or not results[name1]['overall'] or not results[name2]['overall']: print(f"   Cannot compare '{name1}' vs '{name2}': Results missing."); continue

            print(f"\n   **Comparison: {name1} vs {name2}**")
            metrics1 = results[name1]['overall']
            metrics2 = results[name2]['overall']
            max_name_len = max(len(name1), len(name2), 7) # Min width 7 for "Test_B" etc.

            # Header
            print(f"| Metric            | {name1:<{max_name_len}} | {name2:<{max_name_len}} | Change ({name2}-{name1}) |")
            # Separator
            print(f"| :---------------- | :{'-'*(max_name_len-1)} | :{'-'*(max_name_len-1)} | :----------------: |")
            # Data Rows
            kpi_diff = 0.0; kpi_name = 'macro_precision'
            for metric in ['accuracy', kpi_name, 'macro_recall', 'macro_f1']:
                 val1 = metrics1.get(metric, float('nan')); val2 = metrics2.get(metric, float('nan'))
                 change = val2 - val1 if not (np.isnan(val1) or np.isnan(val2)) else float('nan')
                 print(f"| {metric:<17} | {format_metric(val1):>{max_name_len}} | {format_metric(val2):>{max_name_len}} | {format_metric(change, 4):>18} |") # Use helper
                 if metric == kpi_name: kpi_diff = change if not np.isnan(change) else 0.0
            print("\n") # Blank line after table

            # Comparison summary text
            if kpi_diff < -0.03: print(f"   *WARNING:* Potential performance degradation detected ({kpi_name} decreased in '{name2}' vs '{name1}').")
            elif kpi_diff > 0.03: print(f"   *INFO:* Performance ({kpi_name}) notably higher in '{name2}' vs '{name1}'.")
            else: print(f"   *INFO:* Performance ({kpi_name}) similar between '{name1}' and '{name2}'.")

    elif comparisons: print("   'comparisons' argument provided but is not a list.")
    else: print("   No specific comparisons requested.")


    # --- 4. Additional Tests/Suggestions --- (No changes)
    print("\n4. Further Analysis Suggestions:")
    # ... (Suggestions) ...

    print("\n--- Evaluation Complete ---")
    return results


# --- Example Usage ---

# (Keep dummy data generation, ensuring 'date_added' is datetime)
# ...
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
# Ensure date_added covers multiple years for the example
data['date_added'] = pd.to_datetime(pd.Timestamp('2020-01-01') + pd.to_timedelta(np.random.randint(0, 3*365, size=len(data)), unit='D'))
target_map = {i: name for i, name in enumerate(newsgroups.target_names)}
data['RCC_Name'] = data['RCC'].map(target_map)
df_train, df_temp = train_test_split(data, test_size=0.5, random_state=42, stratify=data['RCC'])
df_val, df_temp2 = train_test_split(df_temp, test_size=0.6, random_state=123, stratify=df_temp['RCC'])
df_test, df_holdout = train_test_split(df_temp2, test_size=0.5, random_state=456, stratify=df_temp2['RCC'])
print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}, Holdout size: {len(df_holdout)}")
print(f"Date range in data: {data['date_added'].min()} to {data['date_added'].max()}")

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


# --- Run the evaluation function ---
datasets_to_evaluate = {
    'Validation': df_val,
    'Test': df_test,
    'Holdout': df_holdout
}
# Specify datetime column separately
datetime_col = 'date_added'
# Other metadata columns (if any) go here
other_metadata = ['file_type']

comparisons_to_make = [ ('Validation', 'Test'), ('Test', 'Holdout') ]

evaluation_results = evaluate_precomputed_predictions(
    datasets=datasets_to_evaluate,
    text_col='text',
    target_col='RCC_Name',
    pred_col='predicted_label',
    proba_col='prediction_confidence',
    metadata_cols=other_metadata,              # Pass non-datetime cols here
    datetime_metadata_col=datetime_col,        # Pass datetime col here
    comparisons=comparisons_to_make,
    plot_charts=True, # Keep plots enabled for testing
    dpi=100,          # Standard DPI is fine for most uses
    year_start=None,  # Let the function detect start year
    year_end=None,    # Let the function detect end year
    year_bucket_size=1 # Bucket by single year
)

# Expect Markdown formatted tables in the output now
