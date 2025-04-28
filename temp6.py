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
from typing import Dict, Any

def compare_validation_reports(
    report_base_path: str,
    report_ingested_path: str,
    stats_rtol: float = 1e-5, # Relative tolerance for comparing numeric stats
    stats_atol: float = 1e-8  # Absolute tolerance for comparing numeric stats
) -> Dict[str, Any]:
    """
    Compares two validation report JSON files generated by generate_validation_report.

    Args:
        report_base_path: File path to the JSON report from the base platform.
        report_ingested_path: File path to the JSON report from the ingested platform.
        stats_rtol: Relative tolerance for comparing floating point statistics.
        stats_atol: Absolute tolerance for comparing floating point statistics.


    Returns:
        A dictionary summarizing the comparison results.
    """
    print("--- Comparing Validation Reports ---")
    print(f"Base Report: {report_base_path}")
    print(f"Ingested Report: {report_ingested_path}")

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

    # 1. Compare Shape
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

    # 2. Compare Columns (Order matters here based on list comparison)
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
        column_status = 'FAIL' # Still a failure if order specified but different
        column_detail = "Column names match, but order differs."
        results['warnings'].append("Column order differs between reports.")
        print(f"   FAIL: {column_detail}")
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


    # 3. Compare Data Types
    print("3. Comparing data types...")
    base_dtypes = report_base.get('dtypes', {})
    ingested_dtypes = report_ingested.get('dtypes', {})
    dtype_mismatches = {}
    if not base_dtypes or not ingested_dtypes:
        results['checks']['dtype_match'] = {'status': 'FAIL', 'detail': 'DType info missing.'}
        issues_found = True
        print("   FAIL: DType info missing in one or both reports.")
    elif not cols_match: # Can't reliably compare dtypes if columns don't align
         results['checks']['dtype_match'] = {'status': 'SKIPPED', 'detail': 'Skipped due to column mismatch.'}
         print("   SKIPPED: Cannot compare dtypes due to column mismatch.")
    else:
        # Compare dtypes only for columns present in base report (assuming base is source of truth)
        for col in base_cols:
            b_dtype = base_dtypes.get(col)
            i_dtype = ingested_dtypes.get(col)
            if b_dtype != i_dtype:
                dtype_mismatches[col] = {'base': b_dtype, 'ingested': i_dtype}

        if not dtype_mismatches:
            results['checks']['dtype_match'] = {'status': 'PASS', 'mismatches': {}}
            print("   PASS: Data types match.")
        else:
            issues_found = True
            results['checks']['dtype_match'] = {'status': 'FAIL', 'mismatches': dtype_mismatches}
            print(f"   FAIL: Data types differ: {dtype_mismatches}")

    # 4. Compare Descriptive Statistics
    # This requires careful comparison, especially for floats
    print("4. Comparing descriptive statistics...")
    base_stats = report_base.get('descriptive_stats', {})
    ingested_stats = report_ingested.get('descriptive_stats', {})
    stats_mismatches = {}

    if isinstance(base_stats, dict) and base_stats.get('error') or \
       isinstance(ingested_stats, dict) and ingested_stats.get('error'):
        results['checks']['stats_match'] = {'status': 'ERROR', 'detail': 'Stats calculation failed in one report.'}
        issues_found = True # Treat calculation error as a validation failure
        print("   ERROR: Statistics calculation failed in at least one report.")
    elif not base_stats or not ingested_stats:
        results['checks']['stats_match'] = {'status': 'FAIL', 'detail': 'Stats info missing.'}
        issues_found = True
        print("   FAIL: Statistics info missing in one or both reports.")
    elif not cols_match: # Can't reliably compare stats if columns don't align
         results['checks']['stats_match'] = {'status': 'SKIPPED', 'detail': 'Skipped due to column mismatch.'}
         print("   SKIPPED: Cannot compare stats due to column mismatch.")
    else:
        # Compare stats for columns present in base report
        stats_passed = True
        for col in base_cols:
            b_col_stats = base_stats.get(col, {})
            i_col_stats = ingested_stats.get(col, {})
            col_mismatches = {}

            # Compare common keys found in pandas describe() output
            all_stat_keys = set(b_col_stats.keys()) | set(i_col_stats.keys())

            for stat_key in all_stat_keys:
                b_val = b_col_stats.get(stat_key)
                i_val = i_col_stats.get(stat_key)

                # Handle nulls before attempting comparison
                if pd.isna(b_val) and pd.isna(i_val):
                     continue # Both are null, treat as matching

                # Check if values exist in both reports
                if stat_key not in b_col_stats:
                    col_mismatches[stat_key] = {'base': 'MISSING', 'ingested': i_val}
                    continue
                if stat_key not in i_col_stats:
                     col_mismatches[stat_key] = {'base': b_val, 'ingested': 'MISSING'}
                     continue

                # Try numeric comparison first
                try:
                    b_float = float(b_val)
                    i_float = float(i_val)
                    if not np.isclose(b_float, i_float, rtol=stats_rtol, atol=stats_atol, equal_nan=True):
                        col_mismatches[stat_key] = {'base': b_val, 'ingested': i_val}
                except (ValueError, TypeError):
                    # If not numeric, do exact string comparison
                    if str(b_val) != str(i_val):
                        col_mismatches[stat_key] = {'base': b_val, 'ingested': i_val}

            if col_mismatches:
                stats_mismatches[col] = col_mismatches
                stats_passed = False

        if stats_passed:
            results['checks']['stats_match'] = {'status': 'PASS', 'mismatches': {}}
            print("   PASS: Descriptive statistics match (within tolerance).")
        else:
            issues_found = True
            results['checks']['stats_match'] = {'status': 'FAIL', 'mismatches': stats_mismatches}
            print(f"   FAIL: Descriptive statistics differ: {stats_mismatches}")


    # 5. Compare Null Counts
    print("5. Comparing null counts...")
    base_nulls = report_base.get('null_counts', {})
    ingested_nulls = report_ingested.get('null_counts', {})
    null_mismatches = {}
    if not base_nulls or not ingested_nulls:
        results['checks']['null_count_match'] = {'status': 'FAIL', 'detail': 'Null count info missing.'}
        issues_found = True
        print("   FAIL: Null count info missing in one or both reports.")
    elif not cols_match:
        results['checks']['null_count_match'] = {'status': 'SKIPPED', 'detail': 'Skipped due to column mismatch.'}
        print("   SKIPPED: Cannot compare null counts due to column mismatch.")
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


    # 6. Compare Hashes (Content Check)
    print("6. Comparing hashes...")
    base_hash_info = report_base.get('hashing_info', {})
    ingested_hash_info = report_ingested.get('hashing_info', {})

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
            results['warnings'].append("Hashing performed using different key columns!")
            print("   WARN: Hash comparison might be invalid - different key columns used.")
        if base_hash_info.get('sorted_for_hashing') != ingested_hash_info.get('sorted_for_hashing'):
            results['warnings'].append("Hashing performed with different sorting settings!")
            print("   WARN: Hash comparison might be invalid - different sorting settings used.")

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
        else:
            issues_found = True
            results['checks']['hash_match'] = {
                'status': 'FAIL',
                'detail': 'Aggregate hashes DO NOT match or are missing.',
                'base_hash': base_agg_hash,
                'ingested_hash': ingested_agg_hash
            }
            print(f"   FAIL: Aggregate hashes differ. Base={base_agg_hash}, Ingested={ingested_agg_hash}")
            # Note: Comparing individual row hashes requires them to be stored in the report,
            # which can be very large. Aggregate hash is usually sufficient.

    # --- Final Status ---
    results['overall_status'] = 'FAIL' if issues_found else 'PASS'
    print(f"--- Comparison Complete --- Overall Status: {results['overall_status']} ---")

    return results
