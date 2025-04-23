import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import re # For basic tokenization
from tqdm.notebook import tqdm # Optional: for progress bars on large datasets
import warnings

# Suppress specific warnings if desired (e.g., UserWarnings from Seaborn/Matplotlib)
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning) # Tqdm might raise this

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
def plot_discrete_distribution(data, col_name, df_name, rotation=45, max_categories=40, fixed_width=18): # Increased default max_categories and added fixed_width
    """Plots count and normalized count for a discrete column with fixed width."""
    # Ensure the column exists and is not empty
    if col_name not in data.columns or data[col_name].isnull().all():
        print(f"Skipping plot for '{col_name}' in '{df_name}': Column not found or all values are NaN.")
        return

    # Convert to string to handle mixed types or numerical categories gracefully
    data = data.copy() # Avoid SettingWithCopyWarning
    data[col_name] = data[col_name].astype(str)

    # Limit categories if too many
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

    # --- USE FIXED WIDTH ---
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
        plt.close()
        return

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
         plt.close()
         return

    plt.suptitle(f'Distribution Analysis for: {col_name} in {df_name}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# --- Helper Function for Comparing Discrete Distributions ---
def compare_discrete_distributions(dataframes, col_name, rotation=45, max_categories=40, fixed_width=18): # Increased default max_categories and added fixed_width
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

    # --- USE FIXED WIDTH ---
    # Adjust height slightly if needed for legend space
    plt.figure(figsize=(fixed_width, 7))

    try:
        sns.barplot(x='Category', y='Proportion', hue='DataFrame', data=comparison_df_filtered,
                    order=category_order, palette='viridis')
        plt.title(f'Comparison of Normalized "{col_name}" Distribution Across DataFrames{plot_title_suffix}')
        plt.xlabel(col_name)
        plt.ylabel('Proportion')
        plt.xticks(rotation=rotation, ha='right')
        plt.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend
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
    max_categories_plot=40, # Default max categories increased
    plot_width=18 # Default fixed plot width
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
    """
    # --- Setup ---
    plt.rcParams['figure.dpi'] = high_dpi # Update DPI setting
    print("="*80)
    print("Comprehensive NLP EDA Report")
    print("="*80)
    target_dfs = {} # To store DFs with target column later

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
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percent.round(2)}) # Round percentage here
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
            if not df[target_col].isnull().all(): # Ensure target has non-NA values
                 target_dfs[name] = df # Store for later analysis
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

        # Individual Distributions (Histogram/KDE) - Width fixed implicitly by default rcParams unless overridden
        num_dfs_with_col = sum(1 for df in dataframes.values() if col in df.columns and not df[col].isnull().all())
        if num_dfs_with_col > 0:
            # Keep default width here, height adjusts
            plt.figure(figsize=(12, 5 * num_dfs_with_col))
            plot_index = 1
            for name, df in dataframes.items():
                if col in df.columns and not df[col].isnull().all():
                    plt.subplot(num_dfs_with_col, 1, plot_index)
                    sns.histplot(df[col], kde=True, bins=50)
                    plt.title(f'{name}: Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    # Use log scale for token counts if highly skewed? Optional.
                    # if col == 'number of tokens' and df[col].max() / df[col].median() > 100: # Example condition
                    #     plt.yscale('log')
                    #     plt.ylabel('Frequency (Log Scale)')
                    plot_index += 1
                # ... (handling for missing/all NaN columns) ...
            if plot_index > 1:
                plt.suptitle(f'Histograms/KDE for: {col}', fontsize=16, y=1.0)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.show()
            else:
                plt.close()
        else:
            print(f"Column '{col}' not found or all NaN in all DataFrames. Skipping histograms.")


        # Comparison (Box Plots) - Keep default width? Or make slightly wider? Let's keep default.
        plot_data_boxplot = []
        for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 temp_df = df[[col]].dropna().copy()
                 if not temp_df.empty:
                    temp_df['DataFrame'] = name
                    plot_data_boxplot.append(temp_df)

        if plot_data_boxplot:
             plt.figure(figsize=(10, 6)) # Standard boxplot size seems ok
             combined_df_boxplot = pd.concat(plot_data_boxplot, ignore_index=True)
             sns.boxplot(x='DataFrame', y=col, data=combined_df_boxplot, palette='viridis')
             plt.title(f'Comparison of "{col}" Distribution Across DataFrames')
             plt.xlabel('DataFrame')
             plt.ylabel(col)
             plt.show()

        # --- Distribution by Target Label (CHANGED TO VIOLIN PLOT) ---
        print(f"\n--- Distribution of '{col}' by Target ('{target_col}') ---")
        target_dfs_with_col = {name: df for name, df in target_dfs.items() if col in df.columns and not df[col].isnull().all()}
        if target_dfs_with_col:
            for name, df in target_dfs_with_col.items():
                # Ensure target column is also usable
                if not df[target_col].isnull().all():
                    df_plot = df[[col, target_col]].dropna().copy()
                    if df_plot.empty:
                        print(f"Skipping violin plot for {name}: No overlapping non-NaN data for '{col}' and '{target_col}'.")
                        continue

                    df_plot[target_col] = df_plot[target_col].astype(str) # Ensure target is string for ordering
                    target_order = sorted(df_plot[target_col].unique())

                    # --- USE FIXED WIDTH ---
                    plt.figure(figsize=(plot_width, 7)) # Use fixed width, adjust height maybe
                    sns.violinplot(x=target_col, y=col, data=df_plot, palette='viridis', order=target_order,
                                   scale='width', # Makes violins same width for shape comparison
                                   inner='quartiles' # Show quartiles inside violins
                                  )
                    plt.title(f'{name}: Distribution of "{col}" by "{target_col}" (Violin Plot)')
                    plt.xlabel(target_col)
                    plt.ylabel(col)
                    plt.xticks(rotation=label_rotation, ha='right')

                    # Optional: Add log scale for skewed data like token counts
                    # Check for non-positive values before applying log scale if using
                    is_positive = df_plot[col] > 0
                    if col == 'number of tokens' and is_positive.all() and (df_plot[col].max() / df_plot[col].median() > 50): # Example condition & check
                        plt.yscale('log')
                        plt.ylabel(f"{col} (Log Scale)")
                        # Adjust y-axis ticks for log scale if needed
                        # plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
                        # plt.gca().yaxis.set_minor_formatter(mticker.NullFormatter())
                        print(f"Applied log scale to y-axis for {name} due to distribution skew.")
                    elif col == 'number of tokens' and not is_positive.all():
                         print(f"Note: Log scale not applied for {name} as '{col}' contains non-positive values.")


                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Target column '{target_col}' in DataFrame '{name}' contains only NaN values. Skipping distribution by target plot.")
        else:
             print(f"Could not perform analysis by target for '{col}'. Ensure '{target_col}' and '{col}' exist and are not all NaN in train/test/oot.")


    # --- 4. Metadata Analysis: Datetime Columns ---
    print("\n" + "="*80)
    print("--- 4. Metadata Analysis: Datetime Columns ---")
    for col in specific_meta_datetime:
         print(f"\nAnalyzing: {col}")
         dt_dfs = {}
         # ... (Initial Check & Conversion Attempt logic remains the same) ...
         for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 if pd.api.types.is_datetime64_any_dtype(df[col]):
                     dt_dfs[name] = df.copy() # Work with a copy
                 else:
                     # ... (conversion attempt) ...
                     print(f"Attempting datetime conversion for '{col}' in DataFrame '{name}'...")
                     try:
                         df_copy = df.copy()
                         df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                         if not df_copy[col].isnull().all():
                              dt_dfs[name] = df_copy
                              # needs_conversion[name] = True # Keep track if needed elsewhere
                              print(f"Conversion successful for '{name}:{col}'.")
                         else:
                              print(f"Conversion resulted in all NaNs for '{name}:{col}'. Skipping.")
                     except Exception as e:
                         print(f"Could not convert '{name}:{col}' to datetime: {e}. Skipping.")
             # ... (handling for all NaN / missing column) ...


         # --- Perform Analysis (CHANGED TO BAR CHART) ---
         if dt_dfs:
             for name, df_dt in dt_dfs.items():
                  print(f"\n--- Datetime Analysis for {col} in {name} ---")
                  df_dt['YearMonth'] = df_dt[col].dt.to_period('M')
                  monthly_counts = df_dt['YearMonth'].value_counts().sort_index()

                  if not monthly_counts.empty:
                      # Convert PeriodIndex to string for better bar plot labeling
                      monthly_counts.index = monthly_counts.index.astype(str)

                      # --- USE FIXED WIDTH BAR CHART ---
                      plt.figure(figsize=(plot_width, 6))
                      monthly_counts.plot(kind='bar', color=sns.color_palette('viridis', len(monthly_counts)))
                      plt.title(f'{name}: Document Count per Month ({col})')
                      plt.xlabel('Year-Month')
                      plt.ylabel('Number of Documents')
                      plt.xticks(rotation=label_rotation, ha='right')
                      plt.grid(True, axis='y') # Keep grid on y-axis
                      plt.tight_layout()
                      plt.show()
                  else:
                       print(f"No non-NaN data points found for monthly counts in {name} after potential conversion.")
                  # ... (Optional DayOfWeek/Hour analysis) ...
         else:
             print(f"Column '{col}' not found or could not be used as datetime in any relevant dataframe.")


    # --- 5. Out-of-Vocabulary (OOV) Analysis ---
    print("\n" + "="*80)
    print("--- 5. Out-of-Vocabulary (OOV) Analysis ---")

    if oov_reference_df_name not in dataframes:
        print(f"Error: Reference DataFrame '{oov_reference_df_name}' not found in input.")
    else:
        ref_df = dataframes[oov_reference_df_name]
        if text_col not in ref_df.columns:
             print(f"Error: Text column '{text_col}' not found in reference DataFrame '{oov_reference_df_name}'.")
        elif ref_df[text_col].isnull().all():
             print(f"Error: Text column '{text_col}' in reference DataFrame '{oov_reference_df_name}' contains only NaN values.")
        else:
            print(f"Building vocabulary from '{oov_reference_df_name}' DataFrame ('{text_col}' column)...")
            vocab = Counter()
            total_ref_tokens = 0
            ref_vocab_set = set() # For unique words in ref
            for text in tqdm(ref_df[text_col].dropna(), desc=f"Building Vocab ({oov_reference_df_name})"):
                tokens = basic_tokenizer(text)
                vocab.update(tokens)
                ref_vocab_set.update(tokens) # Add unique tokens
                total_ref_tokens += len(tokens)

            # vocab_set remains the same as ref_vocab_set here
            vocab_set = ref_vocab_set
            print(f"Vocabulary size from '{oov_reference_df_name}': {len(vocab_set)} unique tokens.")
            print(f"Total tokens in '{oov_reference_df_name}': {total_ref_tokens}")

            print("\nCalculating OOV percentages:")
            oov_results = {}
            unique_oov_results = {} # For the new metric

            for name, df in dataframes.items():
                if name == oov_reference_df_name:
                    continue

                if text_col not in df.columns or df[text_col].isnull().all():
                    print(f"Warning: Skipping OOV for '{name}' (text column '{text_col}' missing or all NaN).")
                    continue

                oov_count = 0
                total_tokens = 0
                oov_word_set = set() # Track unique OOV words in this df
                target_word_set = set() # Track all unique words in this df

                for text in tqdm(df[text_col].dropna(), desc=f"Calculating OOV ({name})"):
                    tokens = basic_tokenizer(text)
                    target_word_set.update(tokens) # Add all unique words from this df
                    current_oov_tokens = [token for token in tokens if token not in vocab_set]
                    oov_count += len(current_oov_tokens)
                    oov_word_set.update(current_oov_tokens) # Add unique OOV words
                    total_tokens += len(tokens)

                # Calculate Token-based OOV%
                if total_tokens > 0:
                    oov_percentage = (oov_count / total_tokens) * 100
                    oov_results[name] = oov_percentage
                else:
                    oov_results[name] = np.nan

                # Calculate Unique Word-based OOV%
                if len(target_word_set) > 0:
                    unique_oov_percentage = (len(oov_word_set) / len(target_word_set)) * 100
                    unique_oov_results[name] = unique_oov_percentage
                else:
                     unique_oov_results[name] = np.nan

                # Print results for the current dataframe
                print(f"- {name}:")
                if total_tokens > 0:
                    print(f"  - Total Tokens: {total_tokens}")
                    print(f"  - OOV Tokens (Count): {oov_count}")
                    print(f"  - OOV % (Token-based): {oov_percentage:.2f}%")
                else:
                    print(f"  - No non-NaN text found for token-based OOV.")

                if len(target_word_set) > 0:
                    print(f"  - Total Unique Words: {len(target_word_set)}")
                    print(f"  - OOV Unique Words (Count): {len(oov_word_set)}")
                    print(f"  - OOV % (Unique Word-based): {unique_oov_percentage:.2f}%")
                else:
                    print(f"  - No non-NaN text found for unique word-based OOV.")


            # Plot OOV percentages (Token-based)
            valid_oov_results = {k: v for k, v in oov_results.items() if pd.notna(v)}
            if valid_oov_results:
                plt.figure(figsize=(max(6, len(valid_oov_results)*1.5), 5))
                oov_series = pd.Series(valid_oov_results).sort_values()
                sns.barplot(x=oov_series.index, y=oov_series.values, palette='viridis')
                plt.title(f'OOV Percentage (Token-Based vs. "{oov_reference_df_name}")')
                plt.xlabel('DataFrame')
                plt.ylabel('OOV Percentage (%)')
                plt.xticks(rotation=label_rotation, ha='right')
                plt.tight_layout()
                plt.show()

            # Plot OOV percentages (Unique Word-based)
            valid_unique_oov_results = {k: v for k, v in unique_oov_results.items() if pd.notna(v)}
            if valid_unique_oov_results:
                 plt.figure(figsize=(max(6, len(valid_unique_oov_results)*1.5), 5))
                 unique_oov_series = pd.Series(valid_unique_oov_results).sort_values()
                 sns.barplot(x=unique_oov_series.index, y=unique_oov_series.values, palette='magma') # Different palette
                 plt.title(f'OOV Percentage (Unique Word-Based vs. "{oov_reference_df_name}")')
                 plt.xlabel('DataFrame')
                 plt.ylabel('OOV Percentage (%)')
                 plt.xticks(rotation=label_rotation, ha='right')
                 plt.tight_layout()
                 plt.show()


    # --- 6. Cross-Feature Analysis (Examples) ---
    print("\n" + "="*80)
    print("--- 6. Cross-Feature Analysis (Examples) ---")

    # Example 1: First common discrete vs First common continuous
    col1_example1 = common_meta_discrete[0] if common_meta_discrete else None
    col2_example1 = common_meta_continuous[0] if common_meta_continuous else None

    if col1_example1 and col2_example1:
        print(f"\n--- Analyzing Relationship: '{col1_example1}' vs '{col2_example1}' ---")
        cross_feature_data_ex1 = []
        # ... (data preparation logic remains the same) ...
        for name, df in dataframes.items():
            if col1_example1 in df.columns and col2_example1 in df.columns:
                 if not df[col1_example1].isnull().all() and not df[col2_example1].isnull().all():
                     temp_df = df[[col1_example1, col2_example1]].dropna(subset=[col1_example1, col2_example1]).copy()
                     if not temp_df.empty:
                         temp_df['DataFrame'] = name
                         temp_df[col1_example1] = temp_df[col1_example1].astype(str)
                         cross_feature_data_ex1.append(temp_df)

        if cross_feature_data_ex1:
            combined_cross_df_ex1 = pd.concat(cross_feature_data_ex1, ignore_index=True)
            category_order_ex1 = combined_cross_df_ex1[col1_example1].value_counts().index[:max_categories_plot]

            # --- USE FIXED WIDTH ---
            plt.figure(figsize=(plot_width, 7))
            sns.boxplot(x=col1_example1, y=col2_example1, hue='DataFrame', data=combined_cross_df_ex1,
                        palette='viridis', order=category_order_ex1)
            plt.title(f'Relationship between "{col1_example1}" and "{col2_example1}" across DataFrames (Top {len(category_order_ex1)} Categories)')
            plt.xlabel(col1_example1)
            plt.ylabel(col2_example1)
            plt.xticks(rotation=label_rotation, ha='right')
            plt.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.show()
        else:
             print(f"Not enough valid data to analyze relationship between '{col1_example1}' and '{col2_example1}'.")
    else:
        print("\nSkipping Cross-Feature Example 1: Need at least one common discrete and one common continuous column specified.")


    # Example 2: First common discrete vs Target (RCC)
    col1_example2 = common_meta_discrete[0] if common_meta_discrete else None

    if col1_example2 and target_col:
        print(f"\n--- Analyzing Relationship: '{col1_example2}' vs '{target_col}' ---")
        target_dfs_with_col1_ex2 = {
            name: df for name, df in target_dfs.items()
            if col1_example2 in df.columns and not df[col1_example2].isnull().all() and not df[target_col].isnull().all()
            }

        if target_dfs_with_col1_ex2:
            for name, df in target_dfs_with_col1_ex2.items():
                 df_plot = df[[col1_example2, target_col]].dropna().copy()
                 if df_plot.empty:
                     # ...
                     continue
                 df_plot[col1_example2] = df_plot[col1_example2].astype(str)
                 df_plot[target_col] = df_plot[target_col].astype(str)

                 try:
                    cross_tab = pd.crosstab(df_plot[col1_example2], df_plot[target_col], normalize='index') * 100
                    category_order_ex2 = df_plot[col1_example2].value_counts().index[:max_categories_plot]
                    cross_tab = cross_tab.reindex(category_order_ex2).dropna(how='all')

                    if cross_tab.empty:
                        # ...
                        continue

                    # --- USE FIXED WIDTH ---
                    cross_tab.plot(kind='bar', stacked=True, figsize=(plot_width, 7), colormap='viridis') # Use fixed width
                    plt.title(f'{name}: Proportion of "{target_col}" within each "{col1_example2}" (Top {len(category_order_ex2)} Categories)')
                    plt.xlabel(col1_example2)
                    plt.ylabel('Percentage (%)')
                    plt.xticks(rotation=label_rotation, ha='right')
                    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout(rect=[0, 0, 0.9, 1])
                    plt.show()

                 except Exception as e:
                     print(f"Error generating stacked bar plot for {name} ({col1_example2} vs {target_col}): {e}")
        else:
             print(f"Not enough valid data to analyze relationship between '{col1_example2}' and '{target_col}'.")
    # ... (Skipping messages) ...


    print("\n" + "="*80)
    print("EDA Complete.")
    print("="*80)


# --- Example Usage ---

# (create_dummy_data function remains the same as before)
def create_dummy_data(n_rows, name):
    data = {
        'processed_text': [' '.join(np.random.choice(['worda', 'wordb', 'wordc', 'neword', 'extrastuff', 'anothertoken', f'rare_word_{i%10}'], size=np.random.randint(10, 100))) for i in range(n_rows)],
        'file extension': np.random.choice(['.pdf', '.docx', '.txt', '.xlsx', '.msg', None, '.PDF', '.jpg', '.png', '.zip', '.eml', '.html', '.csv'], size=n_rows, p=[0.30, 0.20, 0.08, 0.08, 0.08, 0.05, 0.05, 0.02,0.02,0.02,0.02,0.04,0.04]),
        'number of tokens': np.random.gamma(2, 1500, size=n_rows).astype(int) + 10, # Skewed distribution
    }
    mask_nan_tokens = np.random.choice([True, False], size=n_rows, p=[0.03, 0.97])
    data['number of tokens'] = np.where(mask_nan_tokens, np.nan, data['number of tokens'])
    data['number of tokens'] = np.where(data['number of tokens'] == 0, 1, data['number of tokens']) # Ensure positive for log scale

    data['token bucket'] = pd.cut(data['number of tokens'], bins=[0, 100, 500, 1000, 5000, 10000, np.inf], labels=['0-100', '101-500', '501-1000', '1001-5000', '5001-10000', '10000+'], right=False)

    if name in ['train', 'test', 'oot']:
        # More classes for RCC demo
        rcc_classes = [f'Class{chr(65+i)}' for i in range(10)] + ['ClassK_rare', 'ClassL_rare']
        rcc_probs = [0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.10, 0.10] # Sums > 1, fixed below
        rcc_probs = np.array(rcc_probs) / sum(rcc_probs) # Normalize probabilities
        data['RCC'] = np.random.choice(rcc_classes, size=n_rows, p=rcc_probs)
        mask_nan_rcc = np.random.choice([True, False], size=n_rows, p=[0.02, 0.98])
        data['RCC'] = np.where(mask_nan_rcc, np.nan, data['RCC'])


    if name in ['oot', 'prod']:
        start_date = pd.to_datetime('2022-01-01')
        end_date = pd.to_datetime('2023-12-31')
        random_dates = start_date + pd.to_timedelta(np.random.randint(0, (end_date - start_date).days + 1, size=n_rows), unit='d')
        random_times = pd.to_timedelta(np.random.randint(0, 24*60*60, size=n_rows), unit='s')
        mask_nat = np.random.choice([True, False], size=n_rows, p=[0.04, 0.96])
        data['FileModifiedTime'] = np.where(mask_nat, pd.NaT, random_dates + random_times)
        mask_str = np.random.choice([True, False], size=n_rows, p=[0.03, 0.97])
        data['FileModifiedTime'] = np.where(mask_str & ~mask_nat, 'Invalid Date String', data['FileModifiedTime'])

        data['LOB'] = np.random.choice(['Finance', 'HR', 'Legal', 'Operations', 'Sales', 'Marketing', 'IT', None], size=n_rows, p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])

    df = pd.DataFrame(data)
    num_nan_rows = int(n_rows * 0.01)
    if num_nan_rows > 0 and len(df) > num_nan_rows:
         nan_indices = np.random.choice(df.index, size=num_nan_rows, replace=False)
         df.loc[nan_indices, :] = np.nan

    return df

# Generate DataFrames
train_df = create_dummy_data(2500, 'train')
test_df = create_dummy_data(2500, 'test')
oot_df = create_dummy_data(500, 'oot')
prod_df = create_dummy_data(5000, 'prod')

# Add specific words for OOV demo
if not train_df.empty and 'processed_text' in train_df.columns:
    first_valid_index = train_df['processed_text'].first_valid_index()
    if first_valid_index is not None:
        train_df.loc[first_valid_index, 'processed_text'] = 'unique_train_word example_word ' + str(train_df.loc[first_valid_index, 'processed_text'])
    # Add OOV words to test/oot/prod
    for df_name, df_ in [('test', test_df), ('oot', oot_df), ('prod', prod_df)]:
         if not df_.empty and 'processed_text' in df_.columns:
             first_valid_idx_other = df_['processed_text'].first_valid_index()
             if first_valid_idx_other is not None:
                 df_.loc[first_valid_idx_other, 'processed_text'] = f'oov_word_{df_name} another_oov ' + str(df_.loc[first_valid_idx_other, 'processed_text'])


all_dataframes = {
    'train': train_df,
    'test': test_df,
    'oot': oot_df,
    'prod': prod_df
}

# Run the EDA function with updated parameters
comprehensive_nlp_eda(
    dataframes=all_dataframes,
    text_col='processed_text',
    target_col='RCC',
    common_meta_discrete=['file extension', 'token bucket'],
    common_meta_continuous=['number of tokens'],
    specific_meta_discrete=['LOB'],
    specific_meta_datetime=['FileModifiedTime'],
    oov_reference_df_name='train',
    high_dpi=120, # Lower DPI for speed if needed
    label_rotation=45,
    max_categories_plot=40, # Allow more categories
    plot_width=20 # Set a wider fixed width for category plots
)
