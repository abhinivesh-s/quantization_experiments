# -*- coding: utf-8 -*-
# --- Imports ---
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any # Added List, Tuple, Any for hints
import warnings

# Suppress specific warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# --- Auxiliary EDA Function (Updated) ---
def auxiliary_eda(
    dataframes: Dict[str, pd.DataFrame],
    text_col: Optional[str] = 'processed_text',
    target_col: Optional[str] = None, # ADDED: Target column for conflict check
    cardinality_threshold: int = 50,
    n_conflict_examples: int = 5 # Number of conflicting text examples to show
):
    """
    Performs auxiliary EDA checks on a dictionary of dataframes.

    Checks performed:
    1.  Full Row Duplicates: Count, percentage, and count after dropping.
    2.  Text Column Duplicates: Count, percentage (of non-null), and unique count.
    3.  Conflicting Targets for Duplicate Text (Train/Test only): Identifies texts
        present multiple times with different target labels.
    4.  All-NaN Columns: Identifies columns containing only missing values.
    5.  Zero Variance Columns: Identifies columns with only one unique non-null value.
    6.  High Cardinality Discrete Columns: Lists discrete columns with unique values exceeding threshold.
    7.  Memory Usage: Reports approximate memory usage per dataframe.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary mapping dataframe names to DataFrames.
        text_col (Optional[str]): Name of the text column to check for duplicates. If None, checks 2 & 3 are skipped.
        target_col (Optional[str]): Name of the target column. Required for the conflicting target check (check 3).
        cardinality_threshold (int): Threshold for identifying high cardinality discrete columns.
        n_conflict_examples (int): Max number of example texts with conflicting labels to print.
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
            n_after_drop = df_len - n_duplicates # Count remaining after dropping duplicates
            print(f"\n1. Full Row Duplicates:")
            print(f"   - Count of Duplicate Rows: {n_duplicates}")
            print(f"   - Percentage Duplicate Rows: {perc_duplicates:.2f}%")
            print(f"   - Count After Dropping Duplicates: {n_after_drop}")
        except Exception as e:
            print(f"\n1. Full Row Duplicates: Error calculating - {e}")

        # 2. Text Column Duplicates
        if text_col and text_col in df.columns:
            try:
                # Process non-NaN values only
                text_series_nonan = df[text_col].dropna()
                if not text_series_nonan.empty:
                    text_series_lower = text_series_nonan.astype(str).str.lower()
                    n_text_duplicates = text_series_lower.duplicated().sum()
                    n_unique_texts = text_series_lower.nunique() # Count of unique non-null texts
                    perc_text_duplicates = (n_text_duplicates / len(text_series_lower)) * 100 if len(text_series_lower) > 0 else 0

                    print(f"\n2. Text Column ('{text_col}') Duplicates (Case-Insensitive, Non-Null Only):")
                    print(f"   - Count of Duplicate Entries: {n_text_duplicates}")
                    print(f"   - Percentage Duplicate Entries: {perc_text_duplicates:.2f}%")
                    print(f"   - Count of Unique Text Entries: {n_unique_texts}")
                else:
                     print(f"\n2. Text Column ('{text_col}') Duplicates: No non-null text entries found.")
            except Exception as e:
                 print(f"\n2. Text Column ('{text_col}') Duplicates: Error calculating - {e}")
        elif text_col:
            print(f"\n2. Text Column ('{text_col}') Duplicates: Column not found.")
        else:
            print("\n2. Text Column Duplicates: No text column specified.")

        # 3. Conflicting Targets for Duplicate Text (Train/Test only)
        if name in ['train', 'test'] and text_col and target_col:
            print(f"\n3. Conflicting Labels for Duplicate Text ('{name}' dataset only):")
            if text_col not in df.columns:
                print(f"   - Skipping: Text column '{text_col}' not found.")
            elif target_col not in df.columns:
                print(f"   - Skipping: Target column '{target_col}' not found.")
            else:
                try:
                    # Identify texts that appear more than once (case-insensitive, non-null)
                    df_text_target = df[[text_col, target_col]].dropna(subset=[text_col])
                    if not df_text_target.empty:
                        df_text_target['text_lower'] = df_text_target[text_col].astype(str).str.lower()
                        text_counts = df_text_target['text_lower'].value_counts()
                        duplicate_texts_list = text_counts[text_counts > 1].index.tolist()

                        if not duplicate_texts_list:
                            print("   - No duplicate text entries found (case-insensitive, non-null).")
                        else:
                            print(f"   - Found {len(duplicate_texts_list)} unique texts appearing more than once.")
                            # Filter original df for these duplicate texts
                            df_duplicates_subset = df_text_target[df_text_target['text_lower'].isin(duplicate_texts_list)]

                            # Group by the lowercase text and check for conflicting targets
                            conflicting_texts = []
                            grouped = df_duplicates_subset.groupby('text_lower')
                            for text_content, group in grouped:
                                if group[target_col].nunique() > 1:
                                    conflicting_texts.append({
                                        'text': text_content,
                                        'labels': sorted(list(group[target_col].dropna().unique()))
                                    })

                            if conflicting_texts:
                                print(f"   - Found {len(conflicting_texts)} texts with CONFLICTING target labels:")
                                for i, conflict in enumerate(conflicting_texts):
                                    if i < n_conflict_examples:
                                         print(f"     - Text (lower): \"{conflict['text'][:100]}...\" -> Labels: {conflict['labels']}")
                                    elif i == n_conflict_examples:
                                         print(f"     - ... (showing first {n_conflict_examples} examples)")
                                         break
                            else:
                                print("   - No duplicate texts found with conflicting target labels.")
                    else:
                         print("   - No non-null text entries to check for conflicts.")

                except Exception as e:
                    print(f"   - Error checking for conflicting labels: {e}")
        elif name in ['train', 'test']:
             print(f"\n3. Conflicting Labels Check: Skipping for '{name}' (text_col or target_col not specified/found).")


        # --- Renumber subsequent checks ---

        # 4. All-NaN Columns
        try:
            all_nan_cols = df.columns[df.isnull().all()].tolist()
            print(f"\n4. All-NaN Columns:") # Renumbered
            if all_nan_cols:
                print(f"   - Found {len(all_nan_cols)}: {', '.join(all_nan_cols)}")
            else:
                print("   - None found.")
        except Exception as e:
             print(f"\n4. All-NaN Columns: Error checking - {e}")

        # 5. Zero Variance Columns
        try:
            zero_variance_cols = []
            numeric_cols = df.select_dtypes(include=np.number).columns
            object_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in numeric_cols:
                # Check after dropping NaNs
                if df[col].dropna().nunique() == 1: zero_variance_cols.append(col)
            for col in object_cols:
                 if df[col].dropna().nunique() == 1: zero_variance_cols.append(col)

            print(f"\n5. Zero Variance Columns (Constant Value):") # Renumbered
            if zero_variance_cols:
                print(f"   - Found {len(zero_variance_cols)} (excluding NaNs): {', '.join(zero_variance_cols)}")
            else:
                print("   - None found.")
        except Exception as e:
            print(f"\n5. Zero Variance Columns: Error checking - {e}")

        # 6. High Cardinality Discrete Columns
        try:
            high_cardinality_cols = {}
            discrete_cols_card = df.select_dtypes(include=['object', 'category']).columns
            for col in discrete_cols_card:
                 n_unique = df[col].dropna().nunique()
                 if n_unique > cardinality_threshold:
                     high_cardinality_cols[col] = n_unique

            print(f"\n6. High Cardinality Discrete Columns (>{cardinality_threshold} unique values):") # Renumbered
            if high_cardinality_cols:
                print(f"   - Found {len(high_cardinality_cols)}:")
                for col, count in high_cardinality_cols.items(): print(f"     - '{col}': {count} unique")
            else:
                print("   - None found.")
        except Exception as e:
            print(f"\n6. High Cardinality Discrete Columns: Error checking - {e}")

        # 7. Memory Usage
        try:
            mem_usage_bytes = df.memory_usage(deep=True).sum()
            if mem_usage_bytes < 1024**2: mem_usage_str = f"{mem_usage_bytes / 1024:.2f} KB"
            elif mem_usage_bytes < 1024**3: mem_usage_str = f"{mem_usage_bytes / (1024**2):.2f} MB"
            else: mem_usage_str = f"{mem_usage_bytes / (1024**3):.2f} GB"
            print(f"\n7. Memory Usage:") # Renumbered
            print(f"   - Approximate total memory: {mem_usage_str}")
        except Exception as e:
             print(f"\n7. Memory Usage: Error calculating - {e}")

    print("\n" + "="*80)
    print("Auxiliary EDA Complete.")
    print("="*80)


# --- Example Function Call ---
# Assume `all_dataframes` dictionary exists from the previous examples

# Example: Create placeholder DataFrames if needed
if 'all_dataframes' not in locals(): # Check if the dictionary exists
     print("Creating placeholder dataframes for auxiliary_eda example...")
     placeholder_data_train = {
         'processed_text': ['Text A Train Document', 'TEXT B ABOUT MODELS', 'Train specific Words here', 'TEXT B ABOUT MODELS', 'Text E All Nan Col', 'TEXT B ABOUT MODELS'],
         'constant_col': [1]*6,
         'high_card_col': [f'val_{i}' for i in range(6)],
         'all_nan': [np.nan]*6,
         'RCC': ['ClassA', 'ClassB', 'ClassA', 'ClassC', 'ClassA', 'ClassB'] # Conflicting labels for 'TEXT B ABOUT MODELS'
     }
     train_df_aux = pd.DataFrame(placeholder_data_train)
     train_df_aux.loc[train_df_aux.index[-2], 'processed_text'] = np.nan # Add a NaN text
     # Add full row duplicate
     train_df_aux_dups = pd.concat([train_df_aux, train_df_aux.iloc[[1]]], ignore_index=True)

     placeholder_data_test = {
         'processed_text': ['Test Text 1', 'Test Text 2', 'Test Text 1', 'Test Text 3', 'Test Text 1'],
         'constant_col': ['Test']*5,
         'high_card_col': [f'tval_{i}' for i in range(5)],
         'all_nan': [np.nan]*5,
         'RCC': ['ClassX', 'ClassY', 'ClassX', 'ClassZ', 'ClassX'] # No conflict here
     }
     test_df_aux = pd.DataFrame(placeholder_data_test)


     prod_df_aux = pd.DataFrame({
         'processed_text': [f'Prod text {i}' for i in range(100)],
         'constant_col': ['ProdVal']*100,
         'high_card_col': [f'prod_{i}' for i in range(100)],
         'all_nan': [np.nan]*100
     })
     # Prod doesn't have RCC

     all_dataframes = {'train': train_df_aux_dups, 'test': test_df_aux, 'prod_small': prod_df_aux}

# Define parameters
TEXT_COL_FOR_DUPS = 'processed_text'
TARGET_COL_FOR_CONFLICTS = 'RCC' # Make sure this matches your target column name
CARDINALITY_THRESH = 50
CONFLICT_EXAMPLES_TO_SHOW = 5

# Call the auxiliary EDA function
auxiliary_eda(
    dataframes=all_dataframes,
    text_col=TEXT_COL_FOR_DUPS,
    target_col=TARGET_COL_FOR_CONFLICTS, # Pass the target column name
    cardinality_threshold=CARDINALITY_THRESH,
    n_conflict_examples=CONFLICT_EXAMPLES_TO_SHOW
)
