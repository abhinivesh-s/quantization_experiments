import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import re # For basic tokenization
from tqdm.notebook import tqdm # Optional: for progress bars on large datasets
import warnings
import math # For calculating number of bins

# Suppress specific warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# --- Plotting Configuration ---
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150 # Set high DPI for clarity (default)
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size (will be overridden often)

# --- Helper Function for Basic Tokenization ---
def basic_tokenizer(text):
    """A simple tokenizer that splits on whitespace and removes basic punctuation."""
    if pd.isna(text):
        return []
    # Convert to lowercase, remove punctuation, split
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.split()

# --- Helper Function for Plotting Discrete Distributions ---
def plot_discrete_distribution(data, col_name, df_name, rotation=45, max_categories=40, fixed_width=18):
    """Plots count and normalized count for a discrete column with fixed width."""
    if col_name not in data.columns or data[col_name].isnull().all():
        print(f"Skipping plot for '{col_name}' in '{df_name}': Column not found or all values are NaN.")
        return
    data = data.copy()
    data[col_name] = data[col_name].astype(str)
    value_counts = data[col_name].value_counts()
    if len(value_counts) > max_categories:
        top_categories = value_counts.nlargest(max_categories).index
        data_filtered = data[data[col_name].isin(top_categories)]
        plot_title_suffix = f' (Top {max_categories})'
        category_order = top_categories
    else:
        data_filtered = data
        plot_title_suffix = ''
        category_order = value_counts.index
    if data_filtered.empty:
         print(f"Skipping plot for '{col_name}' in '{df_name}': No data remains after filtering top categories (or original data was empty/NaN).")
         return
    plt.figure(figsize=(fixed_width, 6))
    # Count Plot
    plt.subplot(1, 2, 1)
    try:
        sns.countplot(x=col_name, data=data_filtered, order=category_order, palette='viridis')
        plt.title(f'{df_name}: Distribution of {col_name}{plot_title_suffix}')
        plt.xlabel(col_name)
        plt.ylabel('Count')
        plt.xticks(rotation=rotation, ha='right')
    except Exception as e:
        print(f"Error plotting countplot for {col_name} in {df_name}: {e}")
        plt.close(); return # Close figure on error
    # Normalized Count Plot
    plt.subplot(1, 2, 2)
    try:
        norm_counts = data_filtered[col_name].value_counts(normalize=True).loc[category_order]
        sns.barplot(x=norm_counts.index, y=norm_counts.values, order=category_order, palette='viridis')
        plt.title(f'{df_name}: Normalized Distribution of {col_name}{plot_title_suffix}')
        plt.xlabel(col_name)
        plt.ylabel('Proportion')
        plt.xticks(rotation=rotation, ha='right')
    except Exception as e:
         print(f"Error plotting normalized barplot for {col_name} in {df_name}: {e}")
         plt.close(); return # Close figure on error
    plt.suptitle(f'Distribution Analysis for: {col_name} in {df_name}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# --- Helper Function for Comparing Discrete Distributions ---
def compare_discrete_distributions(dataframes, col_name, rotation=45, max_categories=40, fixed_width=18):
    """Compares normalized distributions of a discrete column across dataframes with fixed width."""
    comparison_data = []
    all_categories = set()
    valid_dfs = {}
    for name, df in dataframes.items():
        if col_name in df.columns and not df[col_name].isnull().all():
            df_copy = df[[col_name]].copy()
            df_copy[col_name] = df_copy[col_name].astype(str)
            counts = df_copy[col_name].value_counts(normalize=True)
            all_categories.update(counts.index)
            valid_dfs[name] = df_copy
            for category, proportion in counts.items():
                comparison_data.append({'DataFrame': name, 'Category': category, 'Proportion': proportion})
        else:
            print(f"Warning: Column '{col_name}' not found or all NaN in DataFrame '{name}'. Skipping comparison for this DF.")
    if not comparison_data:
        print(f"Column '{col_name}' not found or all NaN in any provided dataframe for comparison.")
        return
    comparison_df = pd.DataFrame(comparison_data)
    category_importance = comparison_df.groupby('Category')['Proportion'].mean().sort_values(ascending=False)
    if len(all_categories) > max_categories:
        top_categories = category_importance.nlargest(max_categories).index
        comparison_df_filtered = comparison_df[comparison_df['Category'].isin(top_categories)]
        plot_title_suffix = f' (Top {max_categories} Overall)'
        category_order = top_categories
    else:
        comparison_df_filtered = comparison_df
        plot_title_suffix = ''
        category_order = category_importance.index
    if comparison_df_filtered.empty:
         print(f"Skipping comparison plot for '{col_name}': No data remains after filtering top categories.")
         return
    plt.figure(figsize=(fixed_width, 7))
    try:
        sns.barplot(x='Category', y='Proportion', hue='DataFrame', data=comparison_df_filtered,
                    order=category_order, palette='viridis')
        plt.title(f'Comparison of Normalized "{col_name}" Distribution Across DataFrames{plot_title_suffix}')
        plt.xlabel(col_name)
        plt.ylabel('Proportion')
        plt.xticks(rotation=rotation, ha='right')
        plt.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()
    except Exception as e:
         print(f"Error plotting comparison barplot for {col_name}: {e}")
         plt.close()

# --- Main EDA Function ---
def comprehensive_nlp_eda(
    dataframes, # Dictionary: {'train': train_df, 'test': test_df, 'oot': oot_df, 'prod': prod_df}
    text_col='processed_text',
    target_col='RCC',
    common_meta_discrete=['file extension', 'token bucket'],
    common_meta_continuous=['number of tokens'],
    specific_meta_discrete=['LOB'],
    specific_meta_datetime=['FileModifiedTime'],
    oov_reference_df_name='train',
    high_dpi=150,
    label_rotation=45,
    max_categories_plot=40,
    plot_width=18,
    year_bucket_threshold=15, # Threshold to start bucketing years
    year_bucket_size=2 # Default size for year buckets (e.g., 2 means 2020-2021, 2022-2023)
):
    """
    Performs comprehensive EDA for multiclass NLP classification on multiple dataframes.

    Args:
        dataframes (dict): Dictionary mapping dataframe names (str) to pandas DataFrames.
        text_col (str): Name of the text column.
        target_col (str): Name of the target label column.
        common_meta_discrete (list): List of discrete metadata columns common to all dfs.
        common_meta_continuous (list): List of continuous metadata columns common to all dfs.
        specific_meta_discrete (list): List of discrete metadata columns specific to some dfs (e.g., oot, prod).
        specific_meta_datetime (list): List of datetime metadata columns specific to some dfs (e.g., oot, prod).
        oov_reference_df_name (str): Name of the dataframe in the dictionary to use for reference vocabulary.
        high_dpi (int): DPI setting for matplotlib figures.
        label_rotation (int): Rotation angle for x-axis labels in plots.
        max_categories_plot (int): Maximum number of categories to display in discrete plots.
        plot_width (int): Fixed width for most plots displaying categories.
        year_bucket_threshold (int): Number of unique years above which bucketing is applied.
        year_bucket_size (int): Size of year buckets when applied.
    """
    # --- Setup ---
    plt.rcParams['figure.dpi'] = high_dpi
    print("="*80)
    print("Comprehensive NLP EDA Report")
    print("="*80)
    target_dfs = {}

    # --- 1. Basic Information ---
    print("\n--- 1. Basic Information ---")
    for name, df in dataframes.items():
        print(f"\n--- DataFrame: {name} ---")
        print(f"Shape: {df.shape}")
        print("\nColumns and Data Types:")
        df.info()
        print("\nMissing Value Counts:")
        missing_counts = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percent.round(2)})
        print(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False))
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            print(f"\nDescriptive Statistics for Numerical Columns ({name}):")
            print(df[numerical_cols].describe())
        else:
             print("\nNo numerical columns found for descriptive statistics.")
        if target_col in df.columns:
            print(f"\nTarget Variable ('{target_col}') Distribution ({name}):")
            target_value_counts = df[target_col].value_counts()
            target_value_counts_norm = df[target_col].value_counts(normalize=True)
            target_dist_df = pd.DataFrame({'Count': target_value_counts, 'Proportion': target_value_counts_norm.round(4)})
            print(target_dist_df)
            if not df[target_col].isnull().all():
                 target_dfs[name] = df
        else:
            print(f"\nTarget Variable ('{target_col}') not found in {name}.")

    # --- 2. Metadata Analysis: Discrete Columns ---
    print("\n" + "="*80)
    print("--- 2. Metadata Analysis: Discrete Columns ---")
    # 2a. Target Column Analysis
    print(f"\n--- Target Column ('{target_col}') Analysis ---")
    if target_dfs:
        for name, df in target_dfs.items():
             plot_discrete_distribution(df, target_col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
        compare_discrete_distributions(target_dfs, target_col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
    else:
        print(f"Target column '{target_col}' not found or all NaN in any dataframe for analysis.")
    # 2b. Common Discrete Metadata Analysis
    print("\n--- Common Discrete Metadata Analysis ---")
    for col in common_meta_discrete:
        print(f"\nAnalyzing: {col}")
        for name, df in dataframes.items():
            if col in df.columns:
                plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
            else:
                 print(f"Column '{col}' not found in DataFrame '{name}'.")
        compare_discrete_distributions(dataframes, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
    # 2c. Specific Discrete Metadata Analysis
    print("\n--- Specific Discrete Metadata Analysis ---")
    for col in specific_meta_discrete:
        print(f"\nAnalyzing: {col}")
        specific_dfs = {name: df for name, df in dataframes.items() if col in df.columns}
        if specific_dfs:
            valid_specific_dfs = {n: d for n, d in specific_dfs.items() if not d[col].isnull().all()}
            if valid_specific_dfs:
                for name, df in valid_specific_dfs.items():
                    plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
                compare_discrete_distributions(valid_specific_dfs, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
            else:
                 print(f"Column '{col}' found but contains only NaN values in relevant dataframes.")
        else:
            print(f"Column '{col}' not found in any relevant dataframe (e.g., OOT, Prod).")

    # --- 3. Metadata Analysis: Continuous Columns ---
    print("\n" + "="*80)
    print("--- 3. Metadata Analysis: Continuous Columns ---")
    for col in common_meta_continuous:
        print(f"\nAnalyzing: {col}")
        # Individual Distributions (Histogram/KDE)
        num_dfs_with_col = sum(1 for df in dataframes.values() if col in df.columns and not df[col].isnull().all())
        if num_dfs_with_col > 0:
            plt.figure(figsize=(12, 5 * num_dfs_with_col))
            plot_index = 1
            for name, df in dataframes.items():
                if col in df.columns and not df[col].isnull().all():
                    plt.subplot(num_dfs_with_col, 1, plot_index)
                    sns.histplot(df[col], kde=True, bins=50)
                    plt.title(f'{name}: Distribution of {col}')
                    plt.xlabel(col); plt.ylabel('Frequency')
                    plot_index += 1
                elif col in df.columns and df[col].isnull().all():
                    print(f"Column '{col}' in DataFrame '{name}' contains only NaN values. Skipping histogram.")
                else:
                     print(f"Column '{col}' not found in DataFrame '{name}'. Skipping histogram.")
            if plot_index > 1:
                plt.suptitle(f'Histograms/KDE for: {col}', fontsize=16, y=1.0); plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()
            else: plt.close()
        else: print(f"Column '{col}' not found or all NaN in all DataFrames. Skipping histograms.")

        # Comparison (Box Plots)
        plot_data_boxplot = []
        for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 temp_df = df[[col]].dropna().copy()
                 if not temp_df.empty: temp_df['DataFrame'] = name; plot_data_boxplot.append(temp_df)
        if plot_data_boxplot:
             plt.figure(figsize=(10, 6)); combined_df_boxplot = pd.concat(plot_data_boxplot, ignore_index=True)
             sns.boxplot(x='DataFrame', y=col, data=combined_df_boxplot, palette='viridis')
             plt.title(f'Comparison of "{col}" Distribution Across DataFrames'); plt.xlabel('DataFrame'); plt.ylabel(col); plt.show()

        # Distribution by Target Label (Violin Plot)
        print(f"\n--- Distribution of '{col}' by Target ('{target_col}') ---")
        target_dfs_with_col = {name: df for name, df in target_dfs.items() if col in df.columns and not df[col].isnull().all()}
        if target_dfs_with_col:
            for name, df in target_dfs_with_col.items():
                if not df[target_col].isnull().all():
                    df_plot = df[[col, target_col]].dropna().copy()
                    if df_plot.empty: print(f"Skipping violin plot for {name}: No overlapping non-NaN data."); continue
                    df_plot[target_col] = df_plot[target_col].astype(str)
                    target_order = sorted(df_plot[target_col].unique())
                    plt.figure(figsize=(plot_width, 7))
                    sns.violinplot(x=target_col, y=col, data=df_plot, palette='viridis', order=target_order, scale='width', inner='quartiles')
                    plt.title(f'{name}: Distribution of "{col}" by "{target_col}" (Violin Plot)'); plt.xlabel(target_col); plt.ylabel(col); plt.xticks(rotation=label_rotation, ha='right')
                    # Optional Log Scale Logic
                    is_positive = (df_plot[col] > 0) if pd.api.types.is_numeric_dtype(df_plot[col]) else pd.Series(False, index=df_plot.index)
                    if is_positive.any() and col == 'number of tokens' and (df_plot.loc[is_positive, col].max() / df_plot.loc[is_positive, col].median() > 50):
                        plt.yscale('log')
                        plt.ylabel(f"{col} (Log Scale)")
                        print(f"Applied log scale to y-axis for {name} due to distribution skew.")
                    elif col == 'number of tokens' and not is_positive.all() and is_positive.any():
                         print(f"Note: Log scale not applied for {name} as '{col}' contains non-positive values.")
                    plt.tight_layout(); plt.show()
                else: print(f"Target column '{target_col}' in DataFrame '{name}' contains only NaN values. Skipping distribution by target plot.")
        else: print(f"Could not perform analysis by target for '{col}'. Ensure '{target_col}' and '{col}' exist and are not all NaN in train/test/oot.")

    # --- 4. Metadata Analysis: Datetime Columns ---
    print("\n" + "="*80)
    print("--- 4. Metadata Analysis: Datetime Columns ---")
    for col in specific_meta_datetime:
         print(f"\nAnalyzing: {col}")
         dt_dfs = {}
         # Initial Check & Conversion Attempt
         for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 if pd.api.types.is_datetime64_any_dtype(df[col]):
                     dt_dfs[name] = df.copy()
                 else:
                     print(f"Attempting datetime conversion for '{col}' in DataFrame '{name}'...")
                     try:
                         df_copy = df.copy()
                         df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                         if not df_copy[col].isnull().all(): dt_dfs[name] = df_copy; print(f"Conversion successful for '{name}:{col}'.")
                         else: print(f"Conversion resulted in all NaNs for '{name}:{col}'. Skipping.")
                     except Exception as e: print(f"Could not convert '{name}:{col}' to datetime: {e}. Skipping.")
             elif col in df.columns and df[col].isnull().all(): print(f"Column '{col}' in DataFrame '{name}' contains only NaN values. Skipping datetime analysis.")

         # Perform Analysis (Yearly or Bucketed Yearly Bar Chart)
         if dt_dfs:
             for name, df_dt in dt_dfs.items():
                  print(f"\n--- Datetime Analysis for {col} in {name} ---")
                  df_dt_nonan = df_dt.dropna(subset=[col]).copy() # Work with non-NaN dates
                  if df_dt_nonan.empty:
                      print(f"No valid datetime values found in '{name}' for '{col}' after dropping NaNs.")
                      continue

                  df_dt_nonan['Year'] = df_dt_nonan[col].dt.year
                  yearly_counts = df_dt_nonan['Year'].value_counts().sort_index()
                  unique_years = yearly_counts.index.astype(int) # Ensure years are int

                  if not yearly_counts.empty:
                      num_years = len(unique_years)
                      plot_title = f'{name}: Document Count'
                      x_label = 'Year'

                      if num_years > year_bucket_threshold:
                          print(f"Number of unique years ({num_years}) exceeds threshold ({year_bucket_threshold}). Bucketing years.")
                          min_year, max_year = unique_years.min(), unique_years.max()
                          # Ensure bucket size is at least 1
                          actual_bucket_size = max(1, year_bucket_size)
                          # Create bins. Ensure the last bin includes max_year.
                          bins = list(range(min_year, max_year + actual_bucket_size, actual_bucket_size))
                          # Adjust the last bin edge if it exactly equals max_year and there's more than one bin
                          if len(bins)>1 and bins[-1] == max_year :
                               bins[-1] = max_year + 1 # Extend slightly to ensure max_year is included in the last bucket
                          elif len(bins)>1 and bins[-1] < max_year : # max_year might not be covered
                               bins.append(max_year + 1)
                          elif len(bins)==1: # Only one bin edge means all fall into one bucket if not handled
                              bins.append(max_year + 1)

                          labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
                          # Handle single year spans within a bucket
                          labels = [f"{bins[i]}" if bins[i+1]-1 == bins[i] else label for i, label in enumerate(labels)]

                          df_dt_nonan['Year Bucket'] = pd.cut(df_dt_nonan['Year'], bins=bins, labels=labels, right=False, include_lowest=True)
                          counts_to_plot = df_dt_nonan['Year Bucket'].value_counts().sort_index()
                          plot_title += f' per {actual_bucket_size}-Year Bucket'
                          x_label = f'{actual_bucket_size}-Year Bucket'
                      else:
                          counts_to_plot = yearly_counts
                          plot_title += ' per Year'

                      # Plotting
                      plt.figure(figsize=(plot_width, 6))
                      counts_to_plot.plot(kind='bar', color=sns.color_palette('viridis', len(counts_to_plot)))
                      plt.title(plot_title + f' ({col})')
                      plt.xlabel(x_label)
                      plt.ylabel('Number of Documents')
                      plt.xticks(rotation=label_rotation, ha='right')
                      plt.grid(True, axis='y')
                      plt.tight_layout()
                      plt.show()

                  else:
                       print(f"No non-NaN data points found for yearly counts in {name} after potential conversion.")
         else:
             print(f"Column '{col}' not found or could not be used as datetime in any relevant dataframe.")

    # --- 5. Out-of-Vocabulary (OOV) Analysis ---
    print("\n" + "="*80)
    print("--- 5. Out-of-Vocabulary (OOV) Analysis ---")
    if oov_reference_df_name not in dataframes:
        print(f"Error: Reference DataFrame '{oov_reference_df_name}' not found in input.")
    else:
        ref_df = dataframes[oov_reference_df_name]
        if text_col not in ref_df.columns: print(f"Error: Text column '{text_col}' not found in reference DataFrame '{oov_reference_df_name}'.")
        elif ref_df[text_col].isnull().all(): print(f"Error: Text column '{text_col}' in reference DataFrame '{oov_reference_df_name}' contains only NaN values.")
        else:
            print(f"Building vocabulary from '{oov_reference_df_name}' DataFrame ('{text_col}' column)...")
            vocab_counter = Counter()
            ref_vocab_set = set()
            total_ref_tokens = 0
            for text in tqdm(ref_df[text_col].dropna(), desc=f"Building Vocab ({oov_reference_df_name})"):
                tokens = basic_tokenizer(text)
                vocab_counter.update(tokens); ref_vocab_set.update(tokens); total_ref_tokens += len(tokens)
            vocab_set = ref_vocab_set # Use the set for OOV checks
            print(f"Vocabulary size from '{oov_reference_df_name}': {len(vocab_set)} unique tokens.")
            print(f"Total tokens in '{oov_reference_df_name}': {total_ref_tokens}")

            print("\nCalculating OOV percentages:")
            oov_results = {}; unique_oov_results = {}
            for name, df in dataframes.items():
                if name == oov_reference_df_name: continue
                if text_col not in df.columns or df[text_col].isnull().all():
                    print(f"Warning: Skipping OOV for '{name}' (text column '{text_col}' missing or all NaN)."); continue

                oov_count = 0; total_tokens = 0; oov_word_set = set(); target_word_set = set()
                for text in tqdm(df[text_col].dropna(), desc=f"Calculating OOV ({name})"):
                    tokens = basic_tokenizer(text)
                    target_word_set.update(tokens)
                    current_oov_tokens = [token for token in tokens if token not in vocab_set]
                    oov_count += len(current_oov_tokens); oov_word_set.update(current_oov_tokens); total_tokens += len(tokens)

                if total_tokens > 0: oov_percentage = (oov_count / total_tokens) * 100; oov_results[name] = oov_percentage
                else: oov_results[name] = np.nan
                if len(target_word_set) > 0: unique_oov_percentage = (len(oov_word_set) / len(target_word_set)) * 100; unique_oov_results[name] = unique_oov_percentage
                else: unique_oov_results[name] = np.nan

                print(f"- {name}:")
                if pd.notna(oov_results.get(name)): print(f"  - Total Tokens: {total_tokens}\n  - OOV Tokens (Count): {oov_count}\n  - OOV % (Token-based): {oov_results[name]:.2f}%")
                else: print(f"  - No non-NaN text found for token-based OOV.")
                if pd.notna(unique_oov_results.get(name)): print(f"  - Total Unique Words: {len(target_word_set)}\n  - OOV Unique Words (Count): {len(oov_word_set)}\n  - OOV % (Unique Word-based): {unique_oov_results[name]:.2f}%")
                else: print(f"  - No non-NaN text found for unique word-based OOV.")

            # Plot OOV percentages (Token-based)
            valid_oov_results = {k: v for k, v in oov_results.items() if pd.notna(v)}
            if valid_oov_results:
                plt.figure(figsize=(max(6, len(valid_oov_results)*1.5), 5)); oov_series = pd.Series(valid_oov_results).sort_values()
                sns.barplot(x=oov_series.index, y=oov_series.values, palette='viridis')
                plt.title(f'OOV Percentage (Token-Based vs. "{oov_reference_df_name}")'); plt.xlabel('DataFrame'); plt.ylabel('OOV Percentage (%)')
                plt.xticks(rotation=label_rotation, ha='right'); plt.tight_layout(); plt.show()
            # Plot OOV percentages (Unique Word-based)
            valid_unique_oov_results = {k: v for k, v in unique_oov_results.items() if pd.notna(v)}
            if valid_unique_oov_results:
                 plt.figure(figsize=(max(6, len(valid_unique_oov_results)*1.5), 5)); unique_oov_series = pd.Series(valid_unique_oov_results).sort_values()
                 sns.barplot(x=unique_oov_series.index, y=unique_oov_series.values, palette='magma')
                 plt.title(f'OOV Percentage (Unique Word-Based vs. "{oov_reference_df_name}")'); plt.xlabel('DataFrame'); plt.ylabel('OOV Percentage (%)')
                 plt.xticks(rotation=label_rotation, ha='right'); plt.tight_layout(); plt.show()

    # --- 6. Cross-Feature Analysis (Examples) ---
    print("\n" + "="*80)
    print("--- 6. Cross-Feature Analysis (Examples) ---")
    # Example 1: First common discrete vs First common continuous
    col1_example1 = common_meta_discrete[0] if common_meta_discrete else None
    col2_example1 = common_meta_continuous[0] if common_meta_continuous else None
    if col1_example1 and col2_example1:
        print(f"\n--- Analyzing Relationship: '{col1_example1}' vs '{col2_example1}' ---")
        cross_feature_data_ex1 = []
        for name, df in dataframes.items():
            if col1_example1 in df.columns and col2_example1 in df.columns:
                 if not df[col1_example1].isnull().all() and not df[col2_example1].isnull().all():
                     temp_df = df[[col1_example1, col2_example1]].dropna(subset=[col1_example1, col2_example1]).copy()
                     if not temp_df.empty: temp_df['DataFrame'] = name; temp_df[col1_example1] = temp_df[col1_example1].astype(str); cross_feature_data_ex1.append(temp_df)
        if cross_feature_data_ex1:
            combined_cross_df_ex1 = pd.concat(cross_feature_data_ex1, ignore_index=True)
            category_order_ex1 = combined_cross_df_ex1[col1_example1].value_counts().index[:max_categories_plot]
            plt.figure(figsize=(plot_width, 7))
            sns.boxplot(x=col1_example1, y=col2_example1, hue='DataFrame', data=combined_cross_df_ex1, palette='viridis', order=category_order_ex1)
            plt.title(f'Relationship between "{col1_example1}" and "{col2_example1}" across DataFrames (Top {len(category_order_ex1)} Categories)'); plt.xlabel(col1_example1); plt.ylabel(col2_example1)
            plt.xticks(rotation=label_rotation, ha='right'); plt.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
        else: print(f"Not enough valid data to analyze relationship between '{col1_example1}' and '{col2_example1}'.")
    else: print("\nSkipping Cross-Feature Example 1: Need at least one common discrete and one common continuous column specified.")

    # Example 2: First common discrete vs Target (RCC)
    col1_example2 = common_meta_discrete[0] if common_meta_discrete else None
    if col1_example2 and target_col:
        print(f"\n--- Analyzing Relationship: '{col1_example2}' vs '{target_col}' ---")
        target_dfs_with_col1_ex2 = {name: df for name, df in target_dfs.items() if col1_example2 in df.columns and not df[col1_example2].isnull().all() and not df[target_col].isnull().all()}
        if target_dfs_with_col1_ex2:
            for name, df in target_dfs_with_col1_ex2.items():
                 df_plot = df[[col1_example2, target_col]].dropna().copy()
                 if df_plot.empty: continue
                 df_plot[col1_example2] = df_plot[col1_example2].astype(str); df_plot[target_col] = df_plot[target_col].astype(str)
                 try:
                    cross_tab = pd.crosstab(df_plot[col1_example2], df_plot[target_col], normalize='index') * 100
                    category_order_ex2 = df_plot[col1_example2].value_counts().index[:max_categories_plot]
                    cross_tab = cross_tab.reindex(category_order_ex2).dropna(how='all')
                    if cross_tab.empty: continue
                    cross_tab.plot(kind='bar', stacked=True, figsize=(plot_width, 7), colormap='viridis')
                    plt.title(f'{name}: Proportion of "{target_col}" within each "{col1_example2}" (Top {len(category_order_ex2)} Categories)'); plt.xlabel(col1_example2); plt.ylabel('Percentage (%)')
                    plt.xticks(rotation=label_rotation, ha='right'); plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
                 except Exception as e: print(f"Error generating stacked bar plot for {name} ({col1_example2} vs {target_col}): {e}")
        else: print(f"Not enough valid data to analyze relationship between '{col1_example2}' and '{target_col}'.")
    elif not col1_example2: print("\nSkipping Cross-Feature Example 2: Need at least one common discrete column specified.")

    print("\n" + "="*80)
    print("EDA Complete.")
    print("="*80)


# --- Example Function Call ---
# Assume you have loaded your data into pandas DataFrames:
# train_df, test_df, oot_df, prod_df

# Example: Load dummy dataframes (replace with your actual loading)
# You would typically load from CSV, database, etc. e.g.,
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
# oot_df = pd.read_csv('oot.csv')
# prod_df = pd.read_csv('prod.csv', parse_dates=['FileModifiedTime']) # Example parsing dates on load
# Ensure 'FileModifiedTime' is datetime if not parsed on load:
# oot_df['FileModifiedTime'] = pd.to_datetime(oot_df['FileModifiedTime'], errors='coerce')
# prod_df['FileModifiedTime'] = pd.to_datetime(prod_df['FileModifiedTime'], errors='coerce')

# Create placeholder DataFrames for the example call to run without error
# --- REPLACE THIS WITH YOUR ACTUAL DATAFRAMES ---
placeholder_data = {'processed_text': ['text a', 'text b'], 'file extension': ['.pdf', '.docx'], 'number of tokens': [100, 200], 'token bucket': ['100-500', '100-500'], 'RCC': ['ClassA', 'ClassB']}
placeholder_data_oot = {'processed_text': ['text c'], 'file extension': ['.txt'], 'number of tokens': [50], 'token bucket': ['0-100'], 'RCC': ['ClassA'], 'FileModifiedTime': [pd.Timestamp('2023-01-15')], 'LOB': ['Finance']}
placeholder_data_prod = {'processed_text': ['text d'], 'file extension': ['.msg'], 'number of tokens': [1000], 'token bucket': ['501-1000'], 'FileModifiedTime': [pd.Timestamp('2022-06-10')], 'LOB': ['HR']}
train_df = pd.DataFrame(placeholder_data)
test_df = pd.DataFrame(placeholder_data)
oot_df = pd.DataFrame(placeholder_data_oot)
prod_df = pd.DataFrame(placeholder_data_prod)
# --- END OF PLACEHOLDER DATA ---


# Combine into the required dictionary structure
all_dataframes = {
    'train': train_df,
    'test': test_df,
    'oot': oot_df,
    'prod': prod_df
}

# Define your column names (adjust if different)
TEXT_COLUMN = 'processed_text'
TARGET_COLUMN = 'RCC'
COMMON_DISCRETE_COLS = ['file extension', 'token bucket']
COMMON_CONTINUOUS_COLS = ['number of tokens']
SPECIFIC_DISCRETE_COLS = ['LOB']
SPECIFIC_DATETIME_COLS = ['FileModifiedTime']
REFERENCE_DF_NAME = 'train'

# Call the comprehensive EDA function
comprehensive_nlp_eda(
    dataframes=all_dataframes,
    text_col=TEXT_COLUMN,
    target_col=TARGET_COLUMN,
    common_meta_discrete=COMMON_DISCRETE_COLS,
    common_meta_continuous=COMMON_CONTINUOUS_COLS,
    specific_meta_discrete=SPECIFIC_DISCRETE_COLS,
    specific_meta_datetime=SPECIFIC_DATETIME_COLS,
    oov_reference_df_name=REFERENCE_DF_NAME,
    high_dpi=120,               # Adjust DPI if needed
    label_rotation=45,
    max_categories_plot=40,     # Max categories to show
    plot_width=20,              # Fixed width for category plots
    year_bucket_threshold=15,   # Start bucketing if more than 15 unique years
    year_bucket_size=2          # Use 2-year buckets
)
