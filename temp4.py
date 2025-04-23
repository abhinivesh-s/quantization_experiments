import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re # For basic tokenization
from tqdm.notebook import tqdm # Optional: for progress bars on large datasets
import warnings

# Suppress specific warnings if desired (e.g., UserWarnings from Seaborn/Matplotlib)
# warnings.filterwarnings('ignore', category=UserWarning)

# --- Plotting Configuration ---
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150 # Set high DPI for clarity (default)
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size

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
def plot_discrete_distribution(data, col_name, df_name, rotation=45, max_categories=30):
    """Plots count and normalized count for a discrete column."""
    # Ensure the column exists and is not empty
    if col_name not in data.columns or data[col_name].isnull().all():
        print(f"Skipping plot for '{col_name}' in '{df_name}': Column not found or all values are NaN.")
        return
        
    # Convert to string to handle mixed types or numerical categories gracefully
    data = data.copy() # Avoid SettingWithCopyWarning
    data[col_name] = data[col_name].astype(str)
        
    plt.figure(figsize=(14, 6))

    # Limit categories if too many
    value_counts = data[col_name].value_counts()
    if len(value_counts) > max_categories:
        top_categories = value_counts.nlargest(max_categories).index
        # Filter data for plotting - handle potential non-existent categories if data changed
        data_filtered = data[data[col_name].isin(top_categories)]
        plot_title_suffix = f' (Top {max_categories})'
        category_order = top_categories # Use the order from value_counts
    else:
        data_filtered = data
        plot_title_suffix = ''
        category_order = value_counts.index # Use the order from value_counts

    if data_filtered.empty:
         print(f"Skipping plot for '{col_name}' in '{df_name}': No data remains after filtering top categories (or original data was empty/NaN).")
         plt.close() # Close the empty figure
         return

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
        # Recalculate normalized counts on the *potentially filtered* data
        norm_counts = data_filtered[col_name].value_counts(normalize=True).loc[category_order] # Ensure order matches countplot
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
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()

# --- Helper Function for Comparing Discrete Distributions ---
def compare_discrete_distributions(dataframes, col_name, rotation=45, max_categories=30):
    """Compares normalized distributions of a discrete column across dataframes."""
    comparison_data = []
    all_categories = set()
    valid_dfs = {} # Keep track of DFs that actually have the column

    # Collect data and identify all categories
    for name, df in dataframes.items():
        if col_name in df.columns and not df[col_name].isnull().all():
            # Convert to string for consistency before counting
            df_copy = df[[col_name]].copy()
            df_copy[col_name] = df_copy[col_name].astype(str)
            counts = df_copy[col_name].value_counts(normalize=True)
            all_categories.update(counts.index)
            valid_dfs[name] = df_copy # Store the copy with string type
            for category, proportion in counts.items():
                comparison_data.append({'DataFrame': name, 'Category': category, 'Proportion': proportion})
        else:
            print(f"Warning: Column '{col_name}' not found or all NaN in DataFrame '{name}'. Skipping comparison for this DF.")

    if not comparison_data:
        print(f"Column '{col_name}' not found or all NaN in any provided dataframe for comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Determine overall category importance (e.g., by average proportion or max proportion)
    # Here using average proportion across DFs where the category exists
    category_importance = comparison_df.groupby('Category')['Proportion'].mean().sort_values(ascending=False)

    # Limit categories if necessary
    if len(all_categories) > max_categories:
        top_categories = category_importance.nlargest(max_categories).index
        comparison_df_filtered = comparison_df[comparison_df['Category'].isin(top_categories)]
        plot_title_suffix = f' (Top {max_categories} Overall)'
        category_order = top_categories # Order by importance
    else:
        comparison_df_filtered = comparison_df
        plot_title_suffix = ''
        category_order = category_importance.index # Order by importance

    if comparison_df_filtered.empty:
         print(f"Skipping comparison plot for '{col_name}': No data remains after filtering top categories.")
         return

    plt.figure(figsize=(max(12, len(category_order) * 0.5), 7)) # Adjust width based on categories
    try:
        sns.barplot(x='Category', y='Proportion', hue='DataFrame', data=comparison_df_filtered,
                    order=category_order, palette='viridis') # Use the determined category order
        plt.title(f'Comparison of Normalized "{col_name}" Distribution Across DataFrames{plot_title_suffix}')
        plt.xlabel(col_name)
        plt.ylabel('Proportion')
        plt.xticks(rotation=rotation, ha='right')
        plt.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
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
    max_categories_plot=30 # Max categories to show in discrete plots
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
        # Use buffer to capture info() output if needed for logs, otherwise just print
        df.info()
        print("\nMissing Value Counts:")
        missing_counts = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percent})
        print(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False))

        # Basic stats for numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            print(f"\nDescriptive Statistics for Numerical Columns ({name}):")
            print(df[numerical_cols].describe())
        else:
             print("\nNo numerical columns found for descriptive statistics.")

        # Target variable distribution (if applicable)
        if target_col in df.columns:
            print(f"\nTarget Variable ('{target_col}') Distribution ({name}):")
            target_value_counts = df[target_col].value_counts()
            target_value_counts_norm = df[target_col].value_counts(normalize=True)
            target_dist_df = pd.DataFrame({'Count': target_value_counts, 'Proportion': target_value_counts_norm})
            print(target_dist_df)
            target_dfs[name] = df # Store for later analysis
        else:
            print(f"\nTarget Variable ('{target_col}') not found in {name}.")


    # --- 2. Metadata Analysis: Discrete Columns ---
    print("\n" + "="*80)
    print("--- 2. Metadata Analysis: Discrete Columns ---")

    # 2a. Target Column Analysis (Train, Test, OOT)
    print(f"\n--- Target Column ('{target_col}') Analysis ---")
    if target_dfs:
        # Individual Distributions
        for name, df in target_dfs.items():
             plot_discrete_distribution(df, target_col, name, rotation=label_rotation, max_categories=max_categories_plot)
        # Comparison Plot
        compare_discrete_distributions(target_dfs, target_col, rotation=label_rotation, max_categories=max_categories_plot)
    else:
        print(f"Target column '{target_col}' not found in any dataframe for analysis.")

    # 2b. Common Discrete Metadata Analysis
    print("\n--- Common Discrete Metadata Analysis ---")
    for col in common_meta_discrete:
        print(f"\nAnalyzing: {col}")
        # Individual Distributions
        for name, df in dataframes.items():
            if col in df.columns:
                plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot)
            else:
                 print(f"Column '{col}' not found in DataFrame '{name}'.")
        # Comparison Plot
        compare_discrete_distributions(dataframes, col, rotation=label_rotation, max_categories=max_categories_plot)

    # 2c. Specific Discrete Metadata Analysis
    print("\n--- Specific Discrete Metadata Analysis ---")
    for col in specific_meta_discrete:
        print(f"\nAnalyzing: {col}")
        specific_dfs = {name: df for name, df in dataframes.items() if col in df.columns}
        if specific_dfs:
            # Individual Distributions
            for name, df in specific_dfs.items():
                plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot)
            # Comparison Plot
            compare_discrete_distributions(specific_dfs, col, rotation=label_rotation, max_categories=max_categories_plot)
        else:
            print(f"Column '{col}' not found in any relevant dataframe (e.g., OOT, Prod).")


    # --- 3. Metadata Analysis: Continuous Columns ---
    print("\n" + "="*80)
    print("--- 3. Metadata Analysis: Continuous Columns ---")

    for col in common_meta_continuous:
        print(f"\nAnalyzing: {col}")

        # Individual Distributions (Histogram/KDE)
        # Determine number of DFs that actually have the column
        num_dfs_with_col = sum(1 for df in dataframes.values() if col in df.columns and not df[col].isnull().all())

        if num_dfs_with_col > 0:
            plt.figure(figsize=(12, 5 * num_dfs_with_col)) # Adjust height based on actual plots needed
            plot_index = 1
            valid_dfs_for_hist = {}
            for name, df in dataframes.items():
                if col in df.columns and not df[col].isnull().all():
                    valid_dfs_for_hist[name] = df
                    plt.subplot(num_dfs_with_col, 1, plot_index)
                    sns.histplot(df[col], kde=True, bins=50)
                    plt.title(f'{name}: Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plot_index += 1
                elif col in df.columns and df[col].isnull().all():
                    print(f"Column '{col}' in DataFrame '{name}' contains only NaN values. Skipping histogram.")
                else:
                    print(f"Column '{col}' not found in DataFrame '{name}'. Skipping histogram.")
            if plot_index > 1: # Check if any plots were actually added
                plt.suptitle(f'Histograms/KDE for: {col}', fontsize=16, y=1.0)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.show()
            else:
                plt.close() # Close the figure if no plots were made
        else:
            print(f"Column '{col}' not found or all NaN in all DataFrames. Skipping histograms.")


        # Comparison (Box Plots)
        plot_data_boxplot = []
        for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 # Add a 'DataFrame' column for hue in seaborn
                 temp_df = df[[col]].dropna().copy() # Drop NaNs for boxplot
                 if not temp_df.empty:
                    temp_df['DataFrame'] = name
                    plot_data_boxplot.append(temp_df)
             # else: No need to print warning again, handled above

        if plot_data_boxplot:
             plt.figure(figsize=(10, 6))
             combined_df_boxplot = pd.concat(plot_data_boxplot, ignore_index=True)
             sns.boxplot(x='DataFrame', y=col, data=combined_df_boxplot, palette='viridis')
             plt.title(f'Comparison of "{col}" Distribution Across DataFrames')
             plt.xlabel('DataFrame')
             plt.ylabel(col)
             plt.show()
        # Don't print warning if no data, already covered by histogram check


        # Distribution by Target Label (Train, Test, OOT)
        print(f"\n--- Distribution of '{col}' by Target ('{target_col}') ---")
        target_dfs_with_col = {name: df for name, df in target_dfs.items() if col in df.columns and not df[col].isnull().all()}
        if target_dfs_with_col:
            for name, df in target_dfs_with_col.items():
                # Ensure target column is also usable
                if not df[target_col].isnull().all():
                    # Get sorted unique target labels for consistent order
                    target_order = sorted(df[target_col].astype(str).unique())
                    plt.figure(figsize=(max(10, len(target_order) * 1.2), 6)) # Adjust width based on num classes
                    sns.boxplot(x=target_col, y=col, data=df, palette='viridis', order=target_order)
                    plt.title(f'{name}: Distribution of "{col}" by "{target_col}"')
                    plt.xlabel(target_col)
                    plt.ylabel(col)
                    plt.xticks(rotation=label_rotation, ha='right')
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
         needs_conversion = {}

         # Initial Check & Conversion Attempt
         for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 if pd.api.types.is_datetime64_any_dtype(df[col]):
                     dt_dfs[name] = df.copy() # Work with a copy
                 else:
                     print(f"Attempting datetime conversion for '{col}' in DataFrame '{name}'...")
                     try:
                         df_copy = df.copy()
                         df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                         if not df_copy[col].isnull().all():
                              dt_dfs[name] = df_copy
                              needs_conversion[name] = True
                              print(f"Conversion successful for '{name}:{col}'.")
                         else:
                              print(f"Conversion resulted in all NaNs for '{name}:{col}'. Skipping.")
                     except Exception as e:
                         print(f"Could not convert '{name}:{col}' to datetime: {e}. Skipping.")
             elif col in df.columns and df[col].isnull().all():
                 print(f"Column '{col}' in DataFrame '{name}' contains only NaN values. Skipping datetime analysis.")
             # No message if column simply doesn't exist

         # Perform Analysis on converted/valid DFs
         if dt_dfs:
             for name, df_dt in dt_dfs.items():
                  print(f"\n--- Datetime Analysis for {col} in {name} ---")
                  df_dt['YearMonth'] = df_dt[col].dt.to_period('M')
                  monthly_counts = df_dt['YearMonth'].value_counts().sort_index()

                  if not monthly_counts.empty:
                      plt.figure(figsize=(15, 6))
                      monthly_counts.plot(kind='line', marker='o', legend=False)
                      plt.title(f'{name}: Document Count per Month ({col})')
                      plt.xlabel('Year-Month')
                      plt.ylabel('Number of Documents')
                      plt.xticks(rotation=label_rotation, ha='right')
                      plt.grid(True, axis='y')
                      plt.tight_layout()
                      plt.show()
                  else:
                       print(f"No non-NaN data points found for monthly counts in {name} after potential conversion.")

                  # Optional: Distribution by Day of Week / Hour (can be noisy)
                  # Add these if needed, using plot_discrete_distribution
                  # df_dt['DayOfWeek'] = df_dt[col].dt.day_name()
                  # df_dt['Hour'] = df_dt[col].dt.hour
                  # plot_discrete_distribution(df_dt, 'DayOfWeek', name, rotation=label_rotation)
                  # plot_discrete_distribution(df_dt, 'Hour', name, rotation=0) # Hours don't need rotation
                  # # Clean up added columns if desired
                  # df_dt.drop(['YearMonth', 'DayOfWeek', 'Hour'], axis=1, inplace=True, errors='ignore')

                  # If conversion happened, optionally update the original dataframe in the input dict
                  # if name in needs_conversion and needs_conversion[name]:
                  #     print(f"Note: DataFrame '{name}' was updated with converted datetime column '{col}'.")
                  #     dataframes[name] = df_dt # Update the original dict entry
         else:
             print(f"Column '{col}' not found or could not be used as datetime in any relevant dataframe (e.g., OOT, Prod).")


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
            # Build vocabulary using the basic tokenizer
            vocab = Counter()
            total_ref_tokens = 0
            # Use tqdm for progress bar on potentially large training set
            for text in tqdm(ref_df[text_col].dropna(), desc=f"Building Vocab ({oov_reference_df_name})"):
                tokens = basic_tokenizer(text)
                vocab.update(tokens)
                total_ref_tokens += len(tokens)

            vocab_set = set(vocab.keys())
            print(f"Vocabulary size from '{oov_reference_df_name}': {len(vocab_set)} unique tokens.")
            print(f"Total tokens in '{oov_reference_df_name}': {total_ref_tokens}")

            print("\nCalculating OOV percentages:")
            oov_results = {}
            for name, df in dataframes.items():
                if name == oov_reference_df_name:
                    continue # Skip the reference dataframe itself

                if text_col not in df.columns:
                    print(f"Warning: Text column '{text_col}' not found in DataFrame '{name}'. Skipping OOV calculation.")
                    continue
                if df[text_col].isnull().all():
                    print(f"Warning: Text column '{text_col}' in DataFrame '{name}' contains only NaN values. Skipping OOV calculation.")
                    continue

                oov_count = 0
                total_tokens = 0
                # Use tqdm for progress bar, especially for prod_df
                for text in tqdm(df[text_col].dropna(), desc=f"Calculating OOV ({name})"):
                    tokens = basic_tokenizer(text)
                    for token in tokens:
                        if token not in vocab_set:
                            oov_count += 1
                    total_tokens += len(tokens)

                if total_tokens > 0:
                    oov_percentage = (oov_count / total_tokens) * 100
                    print(f"- {name}:")
                    print(f"  - Total Tokens: {total_tokens}")
                    print(f"  - OOV Tokens: {oov_count}")
                    print(f"  - OOV Percentage: {oov_percentage:.2f}%")
                    oov_results[name] = oov_percentage
                else:
                    print(f"- {name}: No non-NaN text found to calculate OOV.")
                    oov_results[name] = np.nan

            # Plot OOV percentages
            valid_oov_results = {k: v for k, v in oov_results.items() if pd.notna(v)}
            if valid_oov_results:
                plt.figure(figsize=(max(6, len(valid_oov_results)*1.5), 5)) # Adjust width
                oov_series = pd.Series(valid_oov_results).sort_values()
                sns.barplot(x=oov_series.index, y=oov_series.values, palette='viridis')
                plt.title('OOV Percentage Compared to Training Vocabulary')
                plt.xlabel('DataFrame')
                plt.ylabel('OOV Percentage (%)')
                plt.xticks(rotation=label_rotation, ha='right')
                plt.tight_layout()
                plt.show()
            else:
                print("No valid OOV percentages calculated to plot.")

    # --- 6. Cross-Feature Analysis (Examples) ---
    # Add more as needed based on specific hypotheses
    print("\n" + "="*80)
    print("--- 6. Cross-Feature Analysis (Examples) ---")

    # Dynamically select columns for cross-feature analysis examples
    # Example 1: First common discrete vs First common continuous
    col1_example1 = common_meta_discrete[0] if common_meta_discrete else None
    col2_example1 = common_meta_continuous[0] if common_meta_continuous else None

    if col1_example1 and col2_example1:
        print(f"\n--- Analyzing Relationship: '{col1_example1}' vs '{col2_example1}' ---")
        cross_feature_data_ex1 = []
        for name, df in dataframes.items():
            if col1_example1 in df.columns and col2_example1 in df.columns:
                 # Ensure both columns have some non-NaN data
                 if not df[col1_example1].isnull().all() and not df[col2_example1].isnull().all():
                     temp_df = df[[col1_example1, col2_example1]].dropna(subset=[col1_example1, col2_example1]).copy()
                     if not temp_df.empty:
                         temp_df['DataFrame'] = name
                         temp_df[col1_example1] = temp_df[col1_example1].astype(str) # Convert discrete to string for plotting
                         cross_feature_data_ex1.append(temp_df)
                 #else: # Optional: Add print statement if columns exist but are all NaN
                 #   print(f"Skipping DataFrame '{name}' for Ex1 plot: '{col1_example1}' or '{col2_example1}' contains only NaNs.")
            #else: # Optional: Add print statement if columns are missing
            #     print(f"Skipping DataFrame '{name}' for Ex1 plot: Missing '{col1_example1}' or '{col2_example1}'.")


        if cross_feature_data_ex1:
            combined_cross_df_ex1 = pd.concat(cross_feature_data_ex1, ignore_index=True)
            # Determine order for x-axis (top categories based on overall count)
            category_order_ex1 = combined_cross_df_ex1[col1_example1].value_counts().index[:max_categories_plot] # Use param

            plt.figure(figsize=(max(12, len(category_order_ex1)*0.6), 7)) # Adjust width
            sns.boxplot(x=col1_example1, y=col2_example1, hue='DataFrame', data=combined_cross_df_ex1,
                        palette='viridis', order=category_order_ex1)
            plt.title(f'Relationship between "{col1_example1}" and "{col2_example1}" across DataFrames (Top {len(category_order_ex1)} Categories)')
            plt.xlabel(col1_example1)
            plt.ylabel(col2_example1)
            plt.xticks(rotation=label_rotation, ha='right')
            plt.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
            plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend
            plt.show()
        else:
             print(f"Not enough valid data to analyze relationship between '{col1_example1}' and '{col2_example1}'.")
    else:
        print("\nSkipping Cross-Feature Example 1: Need at least one common discrete and one common continuous column specified.")


    # Example 2: First common discrete vs Target (RCC)
    col1_example2 = common_meta_discrete[0] if common_meta_discrete else None

    if col1_example2 and target_col: # Ensure target col is also specified
        print(f"\n--- Analyzing Relationship: '{col1_example2}' vs '{target_col}' ---")
        # Use target_dfs dictionary defined in Section 1
        target_dfs_with_col1_ex2 = {
            name: df for name, df in target_dfs.items()
            if col1_example2 in df.columns and not df[col1_example2].isnull().all() and not df[target_col].isnull().all()
            }

        if target_dfs_with_col1_ex2:
            for name, df in target_dfs_with_col1_ex2.items():
                 # Create copy to avoid modifying original DF
                 df_plot = df[[col1_example2, target_col]].dropna().copy()
                 if df_plot.empty:
                     print(f"Skipping stacked bar for {name}: No overlapping non-NaN data for '{col1_example2}' and '{target_col}'.")
                     continue

                 df_plot[col1_example2] = df_plot[col1_example2].astype(str)
                 df_plot[target_col] = df_plot[target_col].astype(str)

                 # Calculate proportions for stacked bar chart
                 try:
                    cross_tab = pd.crosstab(df_plot[col1_example2], df_plot[target_col], normalize='index') * 100
                    # Order by overall frequency of the category in col1_example2
                    category_order_ex2 = df_plot[col1_example2].value_counts().index[:max_categories_plot] # Use param
                    cross_tab = cross_tab.reindex(category_order_ex2).dropna(how='all') # Reindex and drop rows that are all NaN after reindexing

                    if cross_tab.empty:
                        print(f"Skipping stacked bar for {name}: Crosstab became empty after filtering/reindexing categories.")
                        continue

                    cross_tab.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
                    plt.title(f'{name}: Proportion of "{target_col}" within each "{col1_example2}" (Top {len(category_order_ex2)} Categories)')
                    plt.xlabel(col1_example2)
                    plt.ylabel('Percentage (%)')
                    plt.xticks(rotation=label_rotation, ha='right')
                    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend
                    plt.show()

                 except Exception as e:
                     print(f"Error generating stacked bar plot for {name} ({col1_example2} vs {target_col}): {e}")


        else:
             print(f"Not enough valid data (need non-NaN '{col1_example2}' and '{target_col}' in the same target dataframe) to analyze relationship.")
    elif not col1_example2:
        print("\nSkipping Cross-Feature Example 2: Need at least one common discrete column specified.")
    # No message needed if target_col is missing, handled earlier


    print("\n" + "="*80)
    print("EDA Complete.")
    print("="*80)


# --- Example Usage ---

# Create dummy dataframes mimicking your structure
def create_dummy_data(n_rows, name):
    data = {
        'processed_text': [' '.join(np.random.choice(['worda', 'wordb', 'wordc', 'neword', 'extrastuff', 'anothertoken'], size=np.random.randint(10, 100))) for _ in range(n_rows)],
        'file extension': np.random.choice(['.pdf', '.docx', '.txt', '.xlsx', '.msg', None, '.PDF'], size=n_rows, p=[0.35, 0.25, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'number of tokens': np.random.randint(10, 5000, size=n_rows),
    }
    # Add some NaNs to number of tokens
    mask_nan_tokens = np.random.choice([True, False], size=n_rows, p=[0.03, 0.97])
    data['number of tokens'] = np.where(mask_nan_tokens, np.nan, data['number of tokens'])

    # Calculate token bucket *after* potential NaNs are added
    data['token bucket'] = pd.cut(data['number of tokens'], bins=[0, 100, 500, 1000, 5000, np.inf], labels=['0-100', '101-500', '501-1000', '1001-5000', '5000+'], right=False)
    # Handle cases where 'number of tokens' was NaN - pd.cut assigns NaN which is fine

    if name in ['train', 'test', 'oot']:
        data['RCC'] = np.random.choice(['ClassA', 'ClassB', 'ClassC', 'ClassD', 'ClassE_rare'], size=n_rows, p=[0.38, 0.28, 0.18, 0.08, 0.08])
        # Add some NaNs to RCC
        mask_nan_rcc = np.random.choice([True, False], size=n_rows, p=[0.02, 0.98])
        data['RCC'] = np.where(mask_nan_rcc, np.nan, data['RCC'])


    if name in ['oot', 'prod']:
         # Generate random dates within the last 2 years
        start_date = pd.to_datetime('2022-01-01')
        end_date = pd.to_datetime('2023-12-31')
        random_dates = start_date + pd.to_timedelta(np.random.randint(0, (end_date - start_date).days + 1, size=n_rows), unit='d')
        random_times = pd.to_timedelta(np.random.randint(0, 24*60*60, size=n_rows), unit='s')
        # Introduce some NaT values
        mask_nat = np.random.choice([True, False], size=n_rows, p=[0.04, 0.96])
        data['FileModifiedTime'] = np.where(mask_nat, pd.NaT, random_dates + random_times)
        # Introduce some non-datetime strings
        mask_str = np.random.choice([True, False], size=n_rows, p=[0.03, 0.97])
        data['FileModifiedTime'] = np.where(mask_str & ~mask_nat, 'Invalid Date String', data['FileModifiedTime']) # Add strings only where not NaT

        data['LOB'] = np.random.choice(['Finance', 'HR', 'Legal', 'Operations', None], size=n_rows, p=[0.23, 0.23, 0.23, 0.23, 0.08])

    df = pd.DataFrame(data)
    # Add some full NaN rows
    num_nan_rows = int(n_rows * 0.01) # 1% fully NaN rows
    nan_indices = np.random.choice(df.index, size=num_nan_rows, replace=False)
    df.loc[nan_indices, :] = np.nan

    return df

# Generate DataFrames (adjust sizes as needed for testing)
# Use slightly smaller sizes for faster demonstration run
train_df = create_dummy_data(2500, 'train')
test_df = create_dummy_data(2500, 'test')
oot_df = create_dummy_data(500, 'oot')
prod_df = create_dummy_data(5000, 'prod') # Keep prod larger but not massive for demo

# Add specific words for OOV demo
if not train_df.empty and 'processed_text' in train_df.columns:
    first_valid_index = train_df['processed_text'].first_valid_index()
    if first_valid_index is not None:
        train_df.loc[first_valid_index, 'processed_text'] = 'unique_train_word ' + str(train_df.loc[first_valid_index, 'processed_text'])
    # Add a fully NaN text example to train if possible
    nan_text_idx = train_df[train_df['processed_text'].isnull()].index
    if not nan_text_idx.empty:
        pass # Already handled by create_dummy_data
    elif len(train_df) > 5:
         train_df.loc[train_df.index[5], 'processed_text'] = np.nan # Force a NaN text

# Combine into dictionary
all_dataframes = {
    'train': train_df,
    'test': test_df,
    'oot': oot_df,
    'prod': prod_df
}

# Run the EDA function
comprehensive_nlp_eda(
    dataframes=all_dataframes,
    text_col='processed_text',
    target_col='RCC',
    common_meta_discrete=['file extension', 'token bucket'], # Ensure these names match your data
    common_meta_continuous=['number of tokens'],        # Ensure these names match your data
    specific_meta_discrete=['LOB'],                     # Ensure these names match your data
    specific_meta_datetime=['FileModifiedTime'],        # Ensure these names match your data
    oov_reference_df_name='train',
    high_dpi=120, # Slightly lower DPI for faster example run
    label_rotation=45,
    max_categories_plot=25 # Limit categories in plots for clarity
)
