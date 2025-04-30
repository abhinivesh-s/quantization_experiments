# Ingestion Validation

import pandas as pd
import numpy as np
import hashlib
import json
from typing import List, Optional, Dict, Any

def generate_validation_report(
    df: pd.DataFrame,
    report_filepath: str,
    hash_columns: Optional[List[str]] = None,
    sort_for_hashing: bool = True
) -> Dict[str, Any]:
    """
    Generates a validation report (metadata, stats, optional hashes) for a DataFrame.

    Args:
        df: The DataFrame to analyze (either base or ingested).
        report_filepath: Path to save the generated JSON report.
        hash_columns: A list of column names to use for creating a unique identifier
                      for each row before hashing. If None, hashing is skipped.
                      These columns should ideally form a unique key.
        sort_for_hashing: If True and hash_columns are provided, sort the DataFrame
                          by hash_columns before generating row hashes. Essential if
                          row order is not guaranteed and you want a comparable
                          aggregate hash.

    Returns:
        A dictionary containing the validation report data. Also saves this
        dictionary as a JSON file to report_filepath.
    """
    report = {}
    print(f"--- Generating Validation Report for DataFrame ---")

    # 1. Shape
    report['shape'] = df.shape
    print(f"Shape: {report['shape']}")

    # 2. Columns
    report['columns'] = list(df.columns)
    print(f"Columns: {report['columns']}")

    # 3. Data Types (convert to string for JSON compatibility)
    report['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    print(f"Data Types: {report['dtypes']}")

    # 4. Basic Statistics (convert to JSON-friendly format)
    print("Calculating descriptive statistics...")
    try:
        # Include 'all' for non-numeric stats, handle potential datetime issues
        desc = df.describe(include='all', datetime_is_numeric=True)
        # Convert numpy types to standard python types for JSON
        report['descriptive_stats'] = json.loads(desc.to_json(orient='columns', default_handler=str))
        print("Descriptive statistics calculated.")
    except Exception as e:
        print(f"WARN: Could not calculate descriptive statistics: {e}")
        report['descriptive_stats'] = {"error": f"Could not calculate: {e}"}

    # 5. Null Counts
    report['null_counts'] = {col: int(df[col].isnull().sum()) for col in df.columns}
    print(f"Null Counts: {report['null_counts']}")

    # 6. Hashing (Optional but recommended for content validation)
    report['hashing_info'] = {
        'hashed': False,
        'hash_columns': hash_columns,
        'sorted_for_hashing': sort_for_hashing if hash_columns else None,
        'aggregate_hash': None,
        # 'row_hashes': [] # Storing all row hashes can be large, often aggregate is enough
    }
    if hash_columns:
        print(f"Calculating hashes using columns: {hash_columns} (Sorting: {sort_for_hashing})...")
        try:
            if not all(col in df.columns for col in hash_columns):
                raise ValueError(f"One or more hash_columns not found in DataFrame: {hash_columns}")

            temp_df = df[hash_columns].copy()

            if sort_for_hashing:
                print("Sorting DataFrame for consistent hashing...")
                temp_df = temp_df.sort_values(by=hash_columns) # Sort only the key columns needed for hashing

            # Define a consistent way to represent a row as a string for hashing
            # Handle NaNs consistently (e.g., replace with a specific string)
            def create_row_string(row):
                return "|".join(str(x) if pd.notna(x) else '<<NaN>>' for x in row)

            print("Generating row strings...")
            row_strings = temp_df.apply(create_row_string, axis=1)

            print("Calculating row hashes (SHA256)...")
            # Calculate hash for each row string
            row_hashes = row_strings.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())

            # Calculate an aggregate hash of all sorted row hashes
            # Sort the individual row hashes first to ensure order doesn't matter
            print("Calculating aggregate hash...")
            all_hashes_sorted_string = "".join(sorted(row_hashes.tolist()))
            aggregate_hash = hashlib.sha256(all_hashes_sorted_string.encode('utf-8')).hexdigest()

            report['hashing_info']['hashed'] = True
            report['hashing_info']['aggregate_hash'] = aggregate_hash
            # report['hashing_info']['row_hashes'] = row_hashes.tolist() # Optional: uncomment if needed, but be mindful of size
            print(f"Aggregate Hash (SHA256): {aggregate_hash}")

        except Exception as e:
            print(f"ERROR: Failed to calculate hashes: {e}")
            report['hashing_info']['error'] = f"Hashing failed: {e}"
            report['hashing_info']['aggregate_hash'] = None # Ensure hash is None on error

    # Save report to JSON
    try:
        with open(report_filepath, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Validation report saved to: {report_filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save report to {report_filepath}: {e}")

    print("--- Report Generation Complete ---")
    return report



import json
import numpy as np
import pandas as pd # Need pandas for isna checks potentially
from typing import Dict, Any

def compare_validation_reports(
    report_base_path: str,
    report_ingested_path: str,
    stats_rtol: float = 1e-5,
    stats_atol: float = 1e-8,
    ignore_top_mismatch_on_freq_match: bool = True,
    normalize_top_comparison: bool = False
) -> Dict[str, Any]:
    """
    Compares two validation report JSON files generated by generate_validation_report.

    Args:
        report_base_path: File path to the JSON report from the base platform.
        report_ingested_path: File path to the JSON report from the ingested platform.
        stats_rtol: Relative tolerance for comparing floating point statistics.
        stats_atol: Absolute tolerance for comparing floating point statistics.
        ignore_top_mismatch_on_freq_match: If True, a mismatch in the 'top' statistic
                                           will be ignored (logged as a warning) if the
                                           corresponding 'freq' statistic matches.
                                           Helpful for handling tie-breaking differences.
        normalize_top_comparison: If True, converts 'top' values to lowercase and
                                  strips whitespace before comparison. Use if hashing
                                  also normalizes these aspects.

    Returns:
        A dictionary summarizing the comparison results.
    """
    print("--- Comparing Validation Reports ---")
    print(f"Base Report: {report_base_path}")
    print(f"Ingested Report: {report_ingested_path}")
    print(f"Settings: ignore_top_mismatch_on_freq_match={ignore_top_mismatch_on_freq_match}, normalize_top_comparison={normalize_top_comparison}")


    try:
        with open(report_base_path, 'r') as f:
            report_base = json.load(f)
        with open(report_ingested_path, 'r') as f:
            report_ingested = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load report files: {e}")
        return {'overall_status': 'ERROR', 'error': f"Failed to load reports: {e}"}

    results = {
        'overall_status': 'PENDING',
        'checks': {},
        'warnings': []
    }
    issues_found = False

    # --- Standard Checks (Shape, Columns, Dtypes, Nulls, Hash) ---
    # (Keep the existing code for these checks exactly as before)
    # ... (Code from previous version for checks 1, 2, 3, 5, 6) ...

    # --- 1. Compare Shape ---
    print("1. Comparing shapes...")
    base_shape = tuple(report_base.get('shape', [None, None])) # Use tuple for direct comparison
    ingested_shape = tuple(report_ingested.get('shape', [None, None]))
    shape_match = base_shape == ingested_shape and base_shape != (None, None)
    results['checks']['shape_match'] = {
        'status': 'PASS' if shape_match else 'FAIL',
        'base': base_shape,
        'ingested': ingested_shape
    }
    if not shape_match:
        issues_found = True
        print(f"   FAIL: Shapes differ. Base={base_shape}, Ingested={ingested_shape}")
    else:
        print("   PASS: Shapes match.")

    # --- 2. Compare Columns ---
    print("2. Comparing columns...")
    base_cols = report_base.get('columns', [])
    ingested_cols = report_ingested.get('columns', [])
    cols_match = base_cols == ingested_cols and base_cols != []
    cols_match_set = set(base_cols) == set(ingested_cols) and base_cols != []
    column_status = 'FAIL'
    column_detail = "Columns mismatch or missing."

    if cols_match:
        column_status = 'PASS'
        column_detail = "Columns match (name and order)."
        print("   PASS: Columns match (name and order).")
    elif cols_match_set:
        issues_found = True
        # Allow check to proceed but mark as fail unless order difference is acceptable
        column_status = 'FAIL' # Or 'WARN' depending on requirements
        column_detail = "Column names match, but order differs."
        results['warnings'].append("Column order differs between reports.")
        print(f"   WARN/FAIL: {column_detail}") # Adjusted print
    else:
        issues_found = True
        print(f"   FAIL: {column_detail}")


    results['checks']['column_match'] = {
        'status': column_status,
        'detail': column_detail,
        'base': base_cols,
        'ingested': ingested_cols,
        'base_only': list(set(base_cols) - set(ingested_cols)),
        'ingested_only': list(set(ingested_cols) - set(base_cols))
    }

    # --- 3. Compare Data Types ---
    print("3. Comparing data types...")
    base_dtypes = report_base.get('dtypes', {})
    ingested_dtypes = report_ingested.get('dtypes', {})
    dtype_mismatches = {}
    can_compare_elements = shape_match and cols_match # Need exact column match for reliable element comparison
    if not base_dtypes or not ingested_dtypes:
        results['checks']['dtype_match'] = {'status': 'FAIL', 'detail': 'DType info missing.'}
        issues_found = True
        print("   FAIL: DType info missing in one or both reports.")
    elif not can_compare_elements:
         results['checks']['dtype_match'] = {'status': 'SKIPPED', 'detail': 'Skipped due to shape or column mismatch.'}
         print("   SKIPPED: Cannot compare dtypes due to prior failures.")
    else:
        for col in base_cols:
            b_dtype = base_dtypes.get(col)
            i_dtype = ingested_dtypes.get(col)
             # Add logic here for nuanced dtype comparison if needed (e.g., int64 vs Int64)
            if b_dtype != i_dtype:
                # Example: Consider nullable and standard ints compatible
                if not ((str(b_dtype).startswith('int') or str(b_dtype).startswith('Int')) and \
                        (str(i_dtype).startswith('int') or str(i_dtype).startswith('Int'))):
                   dtype_mismatches[col] = {'base': b_dtype, 'ingested': i_dtype}
                # Add similar checks for float/Float, bool/boolean if needed

        if not dtype_mismatches:
            results['checks']['dtype_match'] = {'status': 'PASS', 'mismatches': {}}
            print("   PASS: Data types match (or are considered compatible).")
        else:
            issues_found = True
            results['checks']['dtype_match'] = {'status': 'FAIL', 'mismatches': dtype_mismatches}
            print(f"   FAIL: Data types differ: {dtype_mismatches}")

    # --- 4. Compare Descriptive Statistics (MODIFIED) ---
    print("4. Comparing descriptive statistics...")
    base_stats = report_base.get('descriptive_stats', {})
    ingested_stats = report_ingested.get('descriptive_stats', {})
    stats_mismatches = {}
    stats_passed = True # Assume pass initially

    if isinstance(base_stats, dict) and base_stats.get('error') or \
       isinstance(ingested_stats, dict) and ingested_stats.get('error'):
        results['checks']['stats_match'] = {'status': 'ERROR', 'detail': 'Stats calculation failed in one report.'}
        issues_found = True
        stats_passed = False
        print("   ERROR: Statistics calculation failed in at least one report.")
    elif not base_stats or not ingested_stats:
        results['checks']['stats_match'] = {'status': 'FAIL', 'detail': 'Stats info missing.'}
        issues_found = True
        stats_passed = False
        print("   FAIL: Statistics info missing in one or both reports.")
    elif not can_compare_elements:
         results['checks']['stats_match'] = {'status': 'SKIPPED', 'detail': 'Skipped due to shape or column mismatch.'}
         print("   SKIPPED: Cannot compare stats due to prior failures.")
         stats_passed = False # Treat skipped as not passing
    else:
        # Compare stats for columns present in base report
        for col in base_cols:
            b_col_stats = base_stats.get(col, {})
            i_col_stats = ingested_stats.get(col, {})
            col_mismatches = {}

            all_stat_keys = set(b_col_stats.keys()) | set(i_col_stats.keys())

            for stat_key in all_stat_keys:
                b_val = b_col_stats.get(stat_key)
                i_val = i_col_stats.get(stat_key)
                values_differ = False # Flag for this specific stat_key

                # Handle nulls before attempting comparison
                # Treat None, np.nan, pd.NA potentially represented as 'null' in JSON as equivalent
                b_is_null = b_val is None or (isinstance(b_val, float) and np.isnan(b_val))
                i_is_null = i_val is None or (isinstance(i_val, float) and np.isnan(i_val))

                if b_is_null and i_is_null:
                     continue # Both are null, treat as matching

                # Check if values exist in both reports for this key
                if stat_key not in b_col_stats or b_is_null:
                    col_mismatches[stat_key] = {'base': 'MISSING_OR_NULL', 'ingested': i_val}
                    values_differ = True
                elif stat_key not in i_col_stats or i_is_null:
                     col_mismatches[stat_key] = {'base': b_val, 'ingested': 'MISSING_OR_NULL'}
                     values_differ = True
                else:
                     # --- Specific Handling for 'top' ---
                    if stat_key == 'top':
                        b_top = b_val
                        i_top = i_val
                        current_tops_differ = False

                        if normalize_top_comparison:
                            try:
                                # Convert to string, lower, strip (handle non-strings)
                                b_top_norm = str(b_top).lower().strip() if b_top is not None else b_top
                                i_top_norm = str(i_top).lower().strip() if i_top is not None else i_top
                                if b_top_norm != i_top_norm:
                                     current_tops_differ = True
                            except Exception: # Handle potential errors during normalization
                                if str(b_top) != str(i_top): # Fallback to simple string compare
                                     current_tops_differ = True
                        elif str(b_top) != str(i_top): # Standard comparison if not normalizing
                             current_tops_differ = True

                        if current_tops_differ:
                            # Tops differ, now check frequency if requested
                            if ignore_top_mismatch_on_freq_match:
                                b_freq = b_col_stats.get('freq')
                                i_freq = i_col_stats.get('freq')
                                # Compare frequencies (treat as numeric, handle None)
                                freqs_match = False
                                if b_freq is not None and i_freq is not None:
                                    try:
                                        # Use tolerance for freq just in case it was float, though unlikely
                                        if np.isclose(float(b_freq), float(i_freq), rtol=stats_rtol, atol=stats_atol):
                                            freqs_match = True
                                    except (ValueError, TypeError):
                                        if str(b_freq) == str(i_freq): # Fallback comparison
                                            freqs_match = True
                                elif b_freq is None and i_freq is None: # Both missing freq is considered match here
                                    freqs_match = True


                                if freqs_match:
                                    # Frequencies match, log warning and IGNORE mismatch for 'top'
                                    warning_msg = (f"Column '{col}': 'top' mismatch ('{b_top}' vs '{i_top}') "
                                                   f"but 'freq' matches ({b_freq}). Potential tie-breaking difference.")
                                    if warning_msg not in results['warnings']:
                                         results['warnings'].append(warning_msg)
                                    print(f"      WARN: {warning_msg}")
                                    # Don't set values_differ = True
                                else:
                                    # Frequencies also differ (or couldn't be compared), record 'top' mismatch
                                    values_differ = True
                            else:
                                # Not ignoring mismatch, so record it
                                values_differ = True

                    # --- Handling for 'freq' and other stats ---
                    elif stat_key == 'freq': # Ensure freq is compared (often numeric)
                        try:
                            if not np.isclose(float(b_val), float(i_val), rtol=stats_rtol, atol=stats_atol, equal_nan=True):
                                values_differ = True
                        except (ValueError, TypeError): # Fallback if not float convertible
                             if str(b_val) != str(i_val):
                                 values_differ = True
                    else:
                         # Default: Try numeric comparison first, then string
                         try:
                             # Use equal_nan=True for stats like min/max which might be NaN
                             if not np.isclose(float(b_val), float(i_val), rtol=stats_rtol, atol=stats_atol, equal_nan=True):
                                 values_differ = True
                         except (ValueError, TypeError):
                             # If not numeric, do exact string comparison
                             if str(b_val) != str(i_val):
                                 values_differ = True

                # Record mismatch if values differed and weren't handled/ignored
                if values_differ:
                    col_mismatches[stat_key] = {'base': b_val, 'ingested': i_val}


            if col_mismatches:
                stats_mismatches[col] = col_mismatches
                stats_passed = False # Mark overall stats check as failed if any column has mismatches

        if stats_passed:
            results['checks']['stats_match'] = {'status': 'PASS', 'mismatches': {}}
            print("   PASS: Descriptive statistics match (within tolerance and rules).")
        else:
            issues_found = True # Mark overall validation as potentially failed
            results['checks']['stats_match'] = {'status': 'FAIL', 'mismatches': stats_mismatches}
            print(f"   FAIL: Descriptive statistics differ: {stats_mismatches}")
            # Check if the ONLY mismatches were ignored 'top' values
            only_ignored_top = True
            for col, mismatches in stats_mismatches.items():
                 if list(mismatches.keys()) != ['top']: # Check if the only key is 'top'
                     only_ignored_top = False
                     break
                 # Further check if this 'top' mismatch was actually ignored (more complex, relies on warnings)
                 # Simplified: if any recorded mismatch exists here, we failed.
                 # The ignore logic prevents 'top' from being added to mismatch dict when freq matches.

            if not stats_mismatches and results['warnings']: # Passed but had warnings (like ignored top)
                 results['checks']['stats_match']['status'] = 'PASS_WITH_WARNINGS'
                 print("      -> Status set to PASS_WITH_WARNINGS due to ignored mismatches (e.g., 'top' with matching 'freq').")


    # --- 5. Compare Null Counts ---
    print("5. Comparing null counts...")
    base_nulls = report_base.get('null_counts', {})
    ingested_nulls = report_ingested.get('null_counts', {})
    null_mismatches = {}
    if not base_nulls or not ingested_nulls:
        results['checks']['null_count_match'] = {'status': 'FAIL', 'detail': 'Null count info missing.'}
        issues_found = True
        print("   FAIL: Null count info missing in one or both reports.")
    elif not can_compare_elements:
        results['checks']['null_count_match'] = {'status': 'SKIPPED', 'detail': 'Skipped due to shape or column mismatch.'}
        print("   SKIPPED: Cannot compare null counts due to prior failures.")
    else:
        nulls_passed = True
        for col in base_cols:
            b_null = base_nulls.get(col, -1) # Use -1 to indicate missing key
            i_null = ingested_nulls.get(col, -1)
            if b_null != i_null:
                null_mismatches[col] = {'base': b_null, 'ingested': i_null}
                nulls_passed = False

        if nulls_passed:
            results['checks']['null_count_match'] = {'status': 'PASS', 'mismatches': {}}
            print("   PASS: Null counts match.")
        else:
            issues_found = True
            results['checks']['null_count_match'] = {'status': 'FAIL', 'mismatches': null_mismatches}
            print(f"   FAIL: Null counts differ: {null_mismatches}")

    # --- 6. Compare Hashes ---
    print("6. Comparing hashes...")
    base_hash_info = report_base.get('hashing_info', {})
    ingested_hash_info = report_ingested.get('hashing_info', {})
    hash_check_passed = False # Default to false

    if not base_hash_info.get('hashed') or not ingested_hash_info.get('hashed'):
        status = 'SKIPPED'
        detail = "Hashing was not performed or failed in one or both reports."
        if base_hash_info.get('error') or ingested_hash_info.get('error'):
            status = 'ERROR'
            detail = f"Hashing failed. Base error: {base_hash_info.get('error')}, Ingested error: {ingested_hash_info.get('error')}"
            issues_found = True # Treat hashing error as validation failure
        elif not base_hash_info.get('hashed') and not ingested_hash_info.get('hashed'):
             detail = "Hashing was not requested or performed in reports."
             # Not necessarily an issue if hashing wasn't requested
        else:
             detail = "Hashing was not performed or failed in one of the reports."
             issues_found = True # If requested but missing in one, it's an issue

        results['checks']['hash_match'] = {'status': status, 'detail': detail}
        print(f"   {status}: {detail}")
    else:
        # Compare hash settings first
        if base_hash_info.get('hash_columns') != ingested_hash_info.get('hash_columns'):
            warning_msg = "Hashing performed using different key columns!"
            results['warnings'].append(warning_msg)
            print(f"   WARN: {warning_msg} Hash comparison might be invalid.")
        if base_hash_info.get('sorted_for_hashing') != ingested_hash_info.get('sorted_for_hashing'):
             warning_msg = "Hashing performed with different sorting settings!"
             results['warnings'].append(warning_msg)
             print(f"   WARN: {warning_msg} Hash comparison might be invalid.")


        # Compare aggregate hash
        base_agg_hash = base_hash_info.get('aggregate_hash')
        ingested_agg_hash = ingested_hash_info.get('aggregate_hash')

        if base_agg_hash and ingested_agg_hash and base_agg_hash == ingested_agg_hash:
            results['checks']['hash_match'] = {
                'status': 'PASS',
                'detail': 'Aggregate hashes match.',
                'base_hash': base_agg_hash,
                'ingested_hash': ingested_agg_hash
            }
            print("   PASS: Aggregate hashes match.")
            hash_check_passed = True # Explicitly set pass
        else:
            issues_found = True
            results['checks']['hash_match'] = {
                'status': 'FAIL',
                'detail': 'Aggregate hashes DO NOT match or are missing.',
                'base_hash': base_agg_hash,
                'ingested_hash': ingested_agg_hash
            }
            print(f"   FAIL: Aggregate hashes differ. Base={base_agg_hash}, Ingested={ingested_agg_hash}")

    # --- Final Status ---
    # Determine final status - fail if any check failed *unless* it was an explicitly ignored/warned issue
    # We use 'issues_found' which gets set on FAIL statuses or significant errors/skips
    results['overall_status'] = 'FAIL' if issues_found else 'PASS'

    # Refine status if only warnings occurred (e.g. ignored top mismatch)
    if results['overall_status'] == 'PASS' and results['warnings']:
        results['overall_status'] = 'PASS_WITH_WARNINGS'

    print(f"--- Comparison Complete --- Overall Status: {results['overall_status']} ---")

    return results

# Example usage remains the same, but you can now test scenarios with top/freq mismatches
# and control the behavior using the new parameters.

# Example call demonstrating new parameters:
# comparison_results = compare_validation_reports(
#     'base_report.json',
#     'ingested_report.json',
#     ignore_top_mismatch_on_freq_match=True, # Default, ignore top diff if freq matches
#     normalize_top_comparison=False           # Default, use exact top comparison
# )

# Example call to enforce strict top comparison:
# comparison_results_strict = compare_validation_reports(
#     'base_report.json',
#     'ingested_report.json',
#     ignore_top_mismatch_on_freq_match=False # Fail even if freq matches
# )

# Example call to normalize case/whitespace for top:
# comparison_results_norm = compare_validation_reports(
#     'base_report.json',
#     'ingested_report.json',
#     normalize_top_comparison=True # Compare lower().strip() versions of top
# )
