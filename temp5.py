# -*- coding: utf-8 -*-
# --- Imports ---
import pandas as pd
import numpy as np
from typing import Dict, Optional # For type hinting compatibility < 3.9

# --- Helper Function for Basic Tokenization (needed for text duplicate check) ---
def basic_tokenizer_aux(text):
    """
    Basic tokenizer for auxiliary checks. Only performs lowercasing and splitting.
    """
    if pd.isna(text):
        return []
    text = str(text).lower()
    return text.split()

# --- Auxiliary EDA Function ---
def auxiliary_eda(
    dataframes: Dict[str, pd.DataFrame],
    text_col: Optional[str] = 'processed_text', # Column to check for text duplicates
    cardinality_threshold: int = 50 # Threshold for reporting high cardinality columns
):
    """
    Performs auxiliary EDA checks on a dictionary of dataframes.

    Checks performed:
    1.  Full Row Duplicates: Count and percentage of identical rows.
    2.  Text Column Duplicates: Count and percentage based on the specified text_col (lowercase).
    3.  All-NaN Columns: Identifies columns containing only missing values.
    4.  Zero Variance Columns: Identifies columns with only one unique non-null value.
    5.  High Cardinality Discrete Columns: Lists discrete columns with unique values exceeding threshold.
    6.  Memory Usage: Reports approximate memory usage per dataframe.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary mapping dataframe names to DataFrames.
        text_col (Optional[str]): Name of the text column to check for duplicates. If None, this check is skipped.
        cardinality_threshold (int): Threshold for identifying high cardinality discrete columns.
    """
    print("="*80)
    print("Auxiliary EDA Checks")
    print("="*80)

    for name, df in dataframes.items():
        print(f"\n--- DataFrame: {name} (Shape: {df.shape}) ---")
        df_len = len(df)
        if df_len == 0:
            print("DataFrame is empty. Skipping checks.")
            continue

        # 1. Full Row Duplicates
        try:
            n_duplicates = df.duplicated().sum()
            perc_duplicates = (n_duplicates / df_len) * 100 if df_len > 0 else 0
            print(f"\n1. Full Row Duplicates:")
            print(f"   - Count: {n_duplicates}")
            print(f"   - Percentage: {perc_duplicates:.2f}%")
        except Exception as e:
            print(f"\n1. Full Row Duplicates: Error calculating - {e}")

        # 2. Text Column Duplicates
        if text_col and text_col in df.columns:
            try:
                # Use basic lowercasing for comparison
                # Note: This converts NaNs to 'nan' string if not dropped, handle this if needed
                # Let's process non-NaN values only for a cleaner check
                text_series = df[text_col].dropna().astype(str).str.lower()
                if not text_series.empty:
                    n_text_duplicates = text_series.duplicated().sum()
                    # Percentage relative to non-null text entries
                    perc_text_duplicates = (n_text_duplicates / len(text_series)) * 100 if len(text_series) > 0 else 0
                    print(f"\n2. Text Column ('{text_col}') Duplicates (Case-Insensitive):")
                    print(f"   - Count (among non-null entries): {n_text_duplicates}")
                    print(f"   - Percentage (among non-null entries): {perc_text_duplicates:.2f}%")
                else:
                     print(f"\n2. Text Column ('{text_col}') Duplicates: No non-null text entries to check.")
            except Exception as e:
                 print(f"\n2. Text Column ('{text_col}') Duplicates: Error calculating - {e}")
        elif text_col:
            print(f"\n2. Text Column ('{text_col}') Duplicates: Column not found.")
        else:
            print("\n2. Text Column Duplicates: No text column specified.")

        # 3. All-NaN Columns
        try:
            all_nan_cols = df.columns[df.isnull().all()].tolist()
            print(f"\n3. All-NaN Columns:")
            if all_nan_cols:
                print(f"   - Found {len(all_nan_cols)} columns with only NaN values: {', '.join(all_nan_cols)}")
            else:
                print("   - None found.")
        except Exception as e:
             print(f"\n3. All-NaN Columns: Error checking - {e}")

        # 4. Zero Variance Columns
        try:
            zero_variance_cols = []
            # Check numeric and object columns separately
            numeric_cols = df.select_dtypes(include=np.number).columns
            object_cols = df.select_dtypes(include=['object', 'category']).columns

            for col in numeric_cols:
                if df[col].dropna().nunique() == 1:
                    zero_variance_cols.append(col)
            for col in object_cols:
                 if df[col].dropna().nunique() == 1:
                    zero_variance_cols.append(col)

            print(f"\n4. Zero Variance Columns (Constant Value):")
            if zero_variance_cols:
                print(f"   - Found {len(zero_variance_cols)} columns with zero variance (excluding NaNs): {', '.join(zero_variance_cols)}")
            else:
                print("   - None found.")
        except Exception as e:
            print(f"\n4. Zero Variance Columns: Error checking - {e}")

        # 5. High Cardinality Discrete Columns
        try:
            high_cardinality_cols = {}
            # Consider object and category types as discrete for this check
            discrete_cols_card = df.select_dtypes(include=['object', 'category']).columns
            for col in discrete_cols_card:
                 n_unique = df[col].dropna().nunique()
                 if n_unique > cardinality_threshold:
                     high_cardinality_cols[col] = n_unique

            print(f"\n5. High Cardinality Discrete Columns (>{cardinality_threshold} unique values):")
            if high_cardinality_cols:
                print(f"   - Found {len(high_cardinality_cols)} high cardinality columns:")
                for col, count in high_cardinality_cols.items():
                    print(f"     - '{col}': {count} unique values")
            else:
                print("   - None found.")
        except Exception as e:
            print(f"\n5. High Cardinality Discrete Columns: Error checking - {e}")


        # 6. Memory Usage
        try:
            mem_usage_bytes = df.memory_usage(deep=True).sum()
            # Convert to MB or GB for readability
            if mem_usage_bytes < 1024**2: # Less than 1 MB
                 mem_usage_str = f"{mem_usage_bytes / 1024:.2f} KB"
            elif mem_usage_bytes < 1024**3: # Less than 1 GB
                 mem_usage_str = f"{mem_usage_bytes / (1024**2):.2f} MB"
            else:
                 mem_usage_str = f"{mem_usage_bytes / (1024**3):.2f} GB"
            print(f"\n6. Memory Usage:")
            print(f"   - Approximate total memory: {mem_usage_str}")
        except Exception as e:
             print(f"\n6. Memory Usage: Error calculating - {e}")

    print("\n" + "="*80)
    print("Auxiliary EDA Complete.")
    print("="*80)


# --- Example Function Call ---
# Assume `all_dataframes` dictionary exists from the previous examples

# Example: Create placeholder DataFrames if needed
if 'all_dataframes' not in locals(): # Check if the dictionary exists
     print("Creating placeholder dataframes for auxiliary_eda example...")
     placeholder_data = {'processed_text': ['Text A Train Document', 'TEXT B ABOUT MODELS', 'Train specific Words here', 'TEXT B ABOUT MODELS', 'Text E All Nan Col'], 'constant_col': [1]*5, 'high_card_col': [f'val_{i}' for i in range(5)], 'all_nan': [np.nan]*5}
     train_df_aux = pd.DataFrame(placeholder_data)
     train_df_aux.loc[train_df_aux.index[-1], 'processed_text'] = np.nan # Add a NaN text
     train_df_aux_dups = pd.concat([train_df_aux, train_df_aux.iloc[[1,3]]], ignore_index=True) # Add full duplicates

     prod_df_aux = pd.DataFrame({
         'processed_text': [f'Prod text {i}' for i in range(1000)],
         'constant_col': ['ProdVal']*1000,
         'high_card_col': [f'prod_{i}' for i in range(1000)],
         'all_nan': [np.nan]*1000
     })

     all_dataframes = {'train': train_df_aux_dups, 'prod_large': prod_df_aux}

# Define parameters
TEXT_COL_FOR_DUPS = 'processed_text'
CARDINALITY_THRESH = 50

# Call the auxiliary EDA function
auxiliary_eda(
    dataframes=all_dataframes,
    text_col=TEXT_COL_FOR_DUPS,
    cardinality_threshold=CARDINALITY_THRESH
)
