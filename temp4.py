import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re # For basic tokenization
from tqdm.notebook import tqdm # Optional: for progress bars on large datasets

# --- Plotting Configuration ---
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150 # Set high DPI for clarity
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
    plt.figure(figsize=(14, 6))

    # Limit categories if too many
    value_counts = data[col_name].value_counts()
    if len(value_counts) > max_categories:
        top_categories = value_counts.nlargest(max_categories).index
        data_filtered = data[data[col_name].isin(top_categories)]
        plot_title_suffix = f' (Top {max_categories})'
    else:
        data_filtered = data
        plot_title_suffix = ''


    # Count Plot
    plt.subplot(1, 2, 1)
    sns.countplot(x=col_name, data=data_filtered, order=data_filtered[col_name].value_counts().index, palette='viridis')
    plt.title(f'{df_name}: Distribution of {col_name}{plot_title_suffix}')
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.xticks(rotation=rotation, ha='right')

    # Normalized Count Plot
    plt.subplot(1, 2, 2)
    norm_counts = data_filtered[col_name].value_counts(normalize=True)
    sns.barplot(x=norm_counts.index, y=norm_counts.values, order=norm_counts.index, palette='viridis')
    plt.title(f'{df_name}: Normalized Distribution of {col_name}{plot_title_suffix}')
    plt.xlabel(col_name)
    plt.ylabel('Proportion')
    plt.xticks(rotation=rotation, ha='right')

    plt.suptitle(f'Distribution Analysis for: {col_name} in {df_name}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()

# --- Helper Function for Comparing Discrete Distributions ---
def compare_discrete_distributions(dataframes, col_name, rotation=45, max_categories=30):
    """Compares normalized distributions of a discrete column across dataframes."""
    comparison_data = []
    all_categories = set()

    # Collect data and identify all categories
    for name, df in dataframes.items():
        if col_name in df.columns:
            counts = df[col_name].value_counts(normalize=True)
            all_categories.update(counts.index)
            for category, proportion in counts.items():
                comparison_data.append({'DataFrame': name, 'Category': category, 'Proportion': proportion})
        else:
            print(f"Warning: Column '{col_name}' not found in DataFrame '{name}'. Skipping comparison for this DF.")

    if not comparison_data:
        print(f"Column '{col_name}' not found in any provided dataframe for comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Limit categories if necessary
    if len(all_categories) > max_categories:
        # Determine top categories based on average proportion across DFs or total count
        # Here, using average proportion for simplicity
        avg_proportions = comparison_df.groupby('Category')['Proportion'].mean().nlargest(max_categories)
        top_categories = avg_proportions.index
        comparison_df = comparison_df[comparison_df['Category'].isin(top_categories)]
        plot_title_suffix = f' (Top {max_categories})'
    else:
        plot_title_suffix = ''
        top_categories = sorted(list(all_categories), key=lambda x: str(x)) # Sort for consistent plotting order

    plt.figure(figsize=(max(12, len(top_categories) * 0.5), 7)) # Adjust width based on categories
    sns.barplot(x='Category', y='Proportion', hue='DataFrame', data=comparison_df,
                order=top_categories, palette='viridis') # Use the sorted/filtered category order
    plt.title(f'Comparison of Normalized "{col_name}" Distribution Across DataFrames{plot_title_suffix}')
    plt.xlabel(col_name)
    plt.ylabel('Proportion')
    plt.xticks(rotation=rotation, ha='right')
    plt.legend(title='DataFrame')
    plt.tight_layout()
    plt.show()


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
    label_rotation=45
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
    """
    plt.rcParams['figure.dpi'] = high_dpi # Update DPI setting

    print("="*80)
    print("Comprehensive NLP EDA Report")
    print("="*80)

    # --- 1. Basic Information ---
    print("\n--- 1. Basic Information ---")
    for name, df in dataframes.items():
        print(f"\n--- DataFrame: {name} ---")
        print(f"Shape: {df.shape}")
        print("\nColumns and Data Types:")
        print(df.info())
        print("\nMissing Value Counts:")
        missing_counts = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percent})
        print(missing_df[missing_df['Count'] > 0])

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
            print(df[target_col].value_counts())
            print("\nNormalized Target Variable Distribution:")
            print(df[target_col].value_counts(normalize=True))
        else:
            print(f"\nTarget Variable ('{target_col}') not found in {name}.")


    # --- 2. Metadata Analysis: Discrete Columns ---
    print("\n" + "="*80)
    print("--- 2. Metadata Analysis: Discrete Columns ---")

    # 2a. Target Column Analysis (Train, Test, OOT)
    print(f"\n--- Target Column ('{target_col}') Analysis ---")
    target_dfs = {name: df for name, df in dataframes.items() if target_col in df.columns}
    if target_dfs:
        # Individual Distributions
        for name, df in target_dfs.items():
             plot_discrete_distribution(df, target_col, name, rotation=label_rotation)
        # Comparison Plot
        compare_discrete_distributions(target_dfs, target_col, rotation=label_rotation)
    else:
        print(f"Target column '{target_col}' not found in any dataframe for analysis.")

    # 2b. Common Discrete Metadata Analysis
    print("\n--- Common Discrete Metadata Analysis ---")
    for col in common_meta_discrete:
        print(f"\nAnalyzing: {col}")
        # Individual Distributions
        for name, df in dataframes.items():
            if col in df.columns:
                plot_discrete_distribution(df, col, name, rotation=label_rotation)
            else:
                 print(f"Column '{col}' not found in DataFrame '{name}'.")
        # Comparison Plot
        compare_discrete_distributions(dataframes, col, rotation=label_rotation)

    # 2c. Specific Discrete Metadata Analysis
    print("\n--- Specific Discrete Metadata Analysis ---")
    for col in specific_meta_discrete:
        print(f"\nAnalyzing: {col}")
        specific_dfs = {name: df for name, df in dataframes.items() if col in df.columns}
        if specific_dfs:
            # Individual Distributions
            for name, df in specific_dfs.items():
                plot_discrete_distribution(df, col, name, rotation=label_rotation)
            # Comparison Plot
            compare_discrete_distributions(specific_dfs, col, rotation=label_rotation)
        else:
            print(f"Column '{col}' not found in any relevant dataframe (e.g., OOT, Prod).")


    # --- 3. Metadata Analysis: Continuous Columns ---
    print("\n" + "="*80)
    print("--- 3. Metadata Analysis: Continuous Columns ---")

    for col in common_meta_continuous:
        print(f"\nAnalyzing: {col}")

        # Individual Distributions (Histogram/KDE)
        plt.figure(figsize=(12, 5 * len(dataframes)))
        plot_index = 1
        for name, df in dataframes.items():
            if col in df.columns:
                plt.subplot(len(dataframes), 1, plot_index)
                sns.histplot(df[col], kde=True, bins=50)
                plt.title(f'{name}: Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plot_index += 1
            else:
                 print(f"Column '{col}' not found in DataFrame '{name}'.")
        plt.suptitle(f'Histograms/KDE for: {col}', fontsize=16, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

        # Comparison (Box Plots)
        plt.figure(figsize=(10, 6))
        plot_data = []
        for name, df in dataframes.items():
             if col in df.columns:
                 # Add a 'DataFrame' column for hue in seaborn
                 temp_df = df[[col]].copy()
                 temp_df['DataFrame'] = name
                 plot_data.append(temp_df)
             else:
                  print(f"Column '{col}' not found in DataFrame '{name}' for boxplot comparison.")

        if plot_data:
             combined_df = pd.concat(plot_data, ignore_index=True)
             sns.boxplot(x='DataFrame', y=col, data=combined_df, palette='viridis')
             plt.title(f'Comparison of "{col}" Distribution Across DataFrames')
             plt.xlabel('DataFrame')
             plt.ylabel(col)
             plt.show()
        else:
             print(f"No data available to compare '{col}' across dataframes.")


        # Distribution by Target Label (Train, Test, OOT)
        print(f"\n--- Distribution of '{col}' by Target ('{target_col}') ---")
        target_dfs = {name: df for name, df in dataframes.items() if target_col in df.columns and col in df.columns}
        if target_dfs:
            for name, df in target_dfs.items():
                plt.figure(figsize=(max(10, df[target_col].nunique() * 1.5), 6)) # Adjust width based on num classes
                sns.boxplot(x=target_col, y=col, data=df, palette='viridis', order=sorted(df[target_col].unique()))
                plt.title(f'{name}: Distribution of "{col}" by "{target_col}"')
                plt.xlabel(target_col)
                plt.ylabel(col)
                plt.xticks(rotation=label_rotation, ha='right')
                plt.tight_layout()
                plt.show()
        else:
             print(f"Could not perform analysis by target for '{col}'. Ensure '{target_col}' and '{col}' exist in train/test/oot.")


    # --- 4. Metadata Analysis: Datetime Columns ---
    print("\n" + "="*80)
    print("--- 4. Metadata Analysis: Datetime Columns ---")
    for col in specific_meta_datetime:
         print(f"\nAnalyzing: {col}")
         dt_dfs = {name: df for name, df in dataframes.items() if col in df.columns}
         if dt_dfs:
             for name, df in dt_dfs.items():
                 # Ensure column is datetime
                 if pd.api.types.is_datetime64_any_dtype(df[col]):
                     print(f"\n--- Datetime Analysis for {col} in {name} ---")
                     df['YearMonth'] = df[col].dt.to_period('M')
                     monthly_counts = df['YearMonth'].value_counts().sort_index()

                     if not monthly_counts.empty:
                         plt.figure(figsize=(15, 6))
                         monthly_counts.plot(kind='line', marker='o')
                         plt.title(f'{name}: Document Count per Month ({col})')
                         plt.xlabel('Year-Month')
                         plt.ylabel('Number of Documents')
                         plt.xticks(rotation=label_rotation, ha='right')
                         plt.grid(True)
                         plt.tight_layout()
                         plt.show()
                     else:
                          print(f"No data points found for monthly counts in {name}.")

                     # Distribution by Day of Week / Hour (Optional - can be noisy)
                     # df['DayOfWeek'] = df[col].dt.day_name()
                     # df['Hour'] = df[col].dt.hour
                     # plot_discrete_distribution(df, 'DayOfWeek', name, rotation=label_rotation)
                     # plot_discrete_distribution(df, 'Hour', name, rotation=0) # Hours don't need rotation

                 else:
                     print(f"Warning: Column '{col}' in DataFrame '{name}' is not a datetime type. Skipping datetime analysis.")
                     print(f"Attempting conversion for '{name}:{col}'...")
                     try:
                         dataframes[name][col] = pd.to_datetime(df[col], errors='coerce')
                         # Re-run analysis after conversion if needed, or notify user
                         print(f"Conversion successful for '{name}:{col}'. Re-run EDA or manually analyze.")
                     except Exception as e:
                         print(f"Could not convert '{name}:{col}' to datetime: {e}")

         else:
             print(f"Column '{col}' not found in any relevant dataframe (e.g., OOT, Prod).")


    # --- 5. Out-of-Vocabulary (OOV) Analysis ---
    print("\n" + "="*80)
    print("--- 5. Out-of-Vocabulary (OOV) Analysis ---")

    if oov_reference_df_name not in dataframes:
        print(f"Error: Reference DataFrame '{oov_reference_df_name}' not found in input.")
        return

    ref_df = dataframes[oov_reference_df_name]
    if text_col not in ref_df.columns:
         print(f"Error: Text column '{text_col}' not found in reference DataFrame '{oov_reference_df_name}'.")
         return

    print(f"Building vocabulary from '{oov_reference_df_name}' DataFrame ('{text_col}' column)...")
    # Build vocabulary using the basic tokenizer
    vocab = Counter()
    total_ref_tokens = 0
    # Use tqdm for progress bar on potentially large training set
    for text in tqdm(ref_df[text_col], desc=f"Building Vocab ({oov_reference_df_name})"):
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

        oov_count = 0
        total_tokens = 0
        # Use tqdm for progress bar, especially for prod_df
        for text in tqdm(df[text_col], desc=f"Calculating OOV ({name})"):
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
            print(f"- {name}: No tokens found to calculate OOV.")
            oov_results[name] = np.nan

    # Optional: Plot OOV percentages
    if oov_results:
        plt.figure(figsize=(8, 5))
        oov_df = pd.Series(oov_results).sort_values()
        sns.barplot(x=oov_df.index, y=oov_df.values, palette='viridis')
        plt.title('OOV Percentage Compared to Training Vocabulary')
        plt.xlabel('DataFrame')
        plt.ylabel('OOV Percentage (%)')
        plt.xticks(rotation=label_rotation, ha='right')
        plt.tight_layout()
        plt.show()

    # --- 6. Cross-Feature Analysis (Examples) ---
    # Add more as needed based on specific hypotheses
    print("\n" + "="*80)
    print("--- 6. Cross-Feature Analysis (Examples) ---")

    # Example: file extension vs number of tokens
    col1 = 'file extension'
    col2 = 'number of tokens'
    print(f"\n--- Analyzing Relationship: '{col1}' vs '{col2}' ---")
    cross_feature_data = []
    for name, df in dataframes.items():
        if col1 in df.columns and col2 in df.columns:
             temp_df = df[[col1, col2]].copy()
             temp_df['DataFrame'] = name
             cross_feature_data.append(temp_df)
        else:
             print(f"Skipping DataFrame '{name}' (missing '{col1}' or '{col2}')")

    if cross_feature_data:
        combined_cross_df = pd.concat(cross_feature_data, ignore_index=True)
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=col1, y=col2, hue='DataFrame', data=combined_cross_df, palette='viridis',
                    order=combined_cross_df[col1].value_counts().index[:20]) # Limit categories shown if needed
        plt.title(f'Relationship between "{col1}" and "{col2}" across DataFrames')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.xticks(rotation=label_rotation, ha='right')
        plt.legend(title='DataFrame')
        plt.tight_layout()
        plt.show()
    else:
         print("Not enough data to analyze relationship.")


    # Example: file extension vs Target (RCC)
    print(f"\n--- Analyzing Relationship: '{col1}' vs '{target_col}' ---")
    target_dfs_with_col1 = {name: df for name, df in target_dfs.items() if col1 in df.columns}
    if target_dfs_with_col1:
        for name, df in target_dfs_with_col1.items():
             # Calculate proportions for stacked bar chart
             cross_tab = pd.crosstab(df[col1], df[target_col], normalize='index') * 100
             cross_tab = cross_tab.reindex(df[col1].value_counts().index[:20]) # Limit categories shown

             cross_tab.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
             plt.title(f'{name}: Proportion of "{target_col}" within each "{col1}"')
             plt.xlabel(col1)
             plt.ylabel('Percentage (%)')
             plt.xticks(rotation=label_rotation, ha='right')
             plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
             plt.tight_layout()
             plt.show()
    else:
         print(f"Not enough data (need '{col1}' and '{target_col}') to analyze relationship.")


    print("\n" + "="*80)
    print("EDA Complete.")
    print("="*80)


# --- Example Usage ---

# Create dummy dataframes mimicking your structure
def create_dummy_data(n_rows, name):
    data = {
        'processed_text': [' '.join(np.random.choice(['worda', 'wordb', 'wordc', 'neword'], size=np.random.randint(10, 100))) for _ in range(n_rows)],
        'file extension': np.random.choice(['.pdf', '.docx', '.txt', '.xlsx', '.msg'], size=n_rows, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        'number of tokens': np.random.randint(10, 5000, size=n_rows),
    }
    data['token bucket'] = pd.cut(data['number of tokens'], bins=[0, 100, 500, 1000, 5000, np.inf], labels=['0-100', '101-500', '501-1000', '1001-5000', '5000+'], right=False)

    if name in ['train', 'test', 'oot']:
        data['RCC'] = np.random.choice(['ClassA', 'ClassB', 'ClassC', 'ClassD'], size=n_rows, p=[0.4, 0.3, 0.2, 0.1])

    if name in ['oot', 'prod']:
         # Generate random dates within the last 2 years
        start_date = pd.to_datetime('2022-01-01')
        end_date = pd.to_datetime('2023-12-31')
        random_dates = start_date + pd.to_timedelta(np.random.randint(0, (end_date - start_date).days + 1, size=n_rows), unit='d')
        data['FileModifiedTime'] = random_dates + pd.to_timedelta(np.random.randint(0, 24*60*60, size=n_rows), unit='s') # Add random time

        data['LOB'] = np.random.choice(['Finance', 'HR', 'Legal', 'Operations'], size=n_rows, p=[0.25, 0.25, 0.25, 0.25])

    return pd.DataFrame(data)

# Generate DataFrames (adjust sizes as needed for testing)
train_df = create_dummy_data(2500, 'train') # Smaller size for quick testing
test_df = create_dummy_data(2500, 'test')
oot_df = create_dummy_data(500, 'oot')
prod_df = create_dummy_data(5000, 'prod') # Smaller size for quick testing

# Add some specific words to train for OOV demo
train_df['processed_text'][0] = 'unique_train_word ' + train_df['processed_text'][0]

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
    common_meta_discrete=['file extension', 'token bucket'],
    common_meta_continuous=['number of tokens'],
    specific_meta_discrete=['LOB'],
    specific_meta_datetime=['FileModifiedTime'],
    oov_reference_df_name='train',
    high_dpi=120, # Slightly lower DPI for faster example run if needed
    label_rotation=45
)
