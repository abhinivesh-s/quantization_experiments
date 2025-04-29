# -*- coding: utf-8 -*-
# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import re # For basic tokenization
from tqdm.notebook import tqdm # Ensure this import is present
import warnings
import math # For calculating number of bins
import os   # For path operations

# Text Analysis Imports
from sklearn.feature_extraction.text import CountVectorizer
# NLTK for stopwords in TTR
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except ImportError:
    print("NLTK not found. Please install it (`pip install nltk`) and download stopwords (`nltk.download('stopwords')`) to remove stopwords from TTR calculation.")
    STOPWORDS = set() # Use an empty set if nltk is not available

# Typing imports for compatibility with Python < 3.9
from typing import List, Tuple, Optional, Dict, Any

# Suppress specific warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# --- Plotting Configuration ---
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

# --- Helper Function for Basic Tokenization ---
def basic_tokenizer(text):
    """
    Basic tokenizer assuming pre-processing (like punctuation removal) is done.
    Only performs lowercasing and splitting.
    """
    if pd.isna(text):
        return []
    text = str(text).lower()
    return text.split()

# --- Helper Function for Plotting Discrete Distributions ---
def plot_discrete_distribution(data, col_name, df_name, rotation=45, max_categories=40, fixed_width=18, category_order_override: Optional[List[str]] = None):
    """Plots count and normalized count for a discrete column with fixed width, allowing order override."""
    if col_name not in data.columns or data[col_name].isnull().all():
        print(f"Skipping plot for '{col_name}' in '{df_name}': Column not found or all values are NaN.")
        return
    data = data.copy(); data[col_name] = data[col_name].astype(str)
    value_counts = data[col_name].value_counts()
    if category_order_override:
        actual_categories_in_data = value_counts.index.tolist()
        final_category_order = [cat for cat in category_order_override if cat in actual_categories_in_data]
        if len(final_category_order) > max_categories:
             override_freq = value_counts.loc[final_category_order]
             final_category_order = override_freq.nlargest(max_categories).index.tolist()
             final_category_order = [cat for cat in category_order_override if cat in final_category_order]
             plot_title_suffix = f' (Top {max_categories} of Ordered)'
        else:
             plot_title_suffix = ' (Ordered)'
        data_filtered = data[data[col_name].isin(final_category_order)]
    else:
        if len(value_counts) > max_categories:
            final_category_order = value_counts.nlargest(max_categories).index.tolist()
            data_filtered = data[data[col_name].isin(final_category_order)]
            plot_title_suffix = f' (Top {max_categories})'
        else:
            final_category_order = value_counts.index.tolist()
            data_filtered = data; plot_title_suffix = ''
    if data_filtered.empty: print(f"Skipping plot for '{col_name}' in '{df_name}': No data remains after filtering."); return
    fig, axes = plt.subplots(1, 2, figsize=(fixed_width, 6))
    try: # Count Plot
        sns.countplot(x=col_name, data=data_filtered, order=final_category_order, palette='viridis', ax=axes[0])
        axes[0].set_title(f'{df_name}: Distribution of {col_name}{plot_title_suffix}'); axes[0].set_xlabel(col_name); axes[0].set_ylabel('Count');
        axes[0].tick_params(axis='x', rotation=rotation, labelsize='small')
        if rotation != 0: plt.setp(axes[0].get_xticklabels(), ha='right', rotation_mode='anchor')
    except Exception as e: print(f"Error plotting countplot for {col_name} in {df_name}: {e}"); plt.close(fig); return
    try: # Normalized Count Plot
        norm_counts_filtered = data_filtered[col_name].value_counts(normalize=True).reindex(final_category_order, fill_value=0)
        sns.barplot(x=norm_counts_filtered.index, y=norm_counts_filtered.values, order=final_category_order, palette='viridis', ax=axes[1])
        axes[1].set_title(f'{df_name}: Normalized Distribution of {col_name}{plot_title_suffix}'); axes[1].set_xlabel(col_name); axes[1].set_ylabel('Proportion');
        axes[1].tick_params(axis='x', rotation=rotation, labelsize='small')
        if rotation != 0: plt.setp(axes[1].get_xticklabels(), ha='right', rotation_mode='anchor')
    except Exception as e: print(f"Error plotting normalized barplot for {col_name} in {df_name}: {e}"); plt.close(fig); return
    fig.suptitle(f'Distribution Analysis for: {col_name} in {df_name}', fontsize=16, y=1.02); fig.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()

# --- Helper Function for Comparing Discrete Distributions ---
def compare_discrete_distributions(dataframes, col_name, rotation=45, max_categories=40, fixed_width=18, category_order_override: Optional[List[str]] = None):
    """Compares normalized distributions of a discrete column across dataframes with fixed width, allowing order override."""
    comparison_data = []; all_categories_present = set(); valid_dfs = {}
    for name, df in dataframes.items():
        if col_name in df.columns and not df[col_name].isnull().all():
            df_copy = df[[col_name]].copy(); df_copy[col_name] = df_copy[col_name].astype(str)
            counts = df_copy[col_name].value_counts(normalize=True); all_categories_present.update(counts.index); valid_dfs[name] = df_copy
            for category, proportion in counts.items(): comparison_data.append({'DataFrame': name, 'Category': category, 'Proportion': proportion})
        else: print(f"Warning: Column '{col_name}' not found or all NaN in DataFrame '{name}'. Skipping.")
    if not comparison_data or len(valid_dfs) < 2:
        print(f"Skipping comparison plot for '{col_name}': Not found in enough dataframes or data is all NaN.")
        return
    comparison_df = pd.DataFrame(comparison_data)
    plot_title_suffix = ''
    if category_order_override:
        final_category_order = [cat for cat in category_order_override if cat in all_categories_present]
        if len(final_category_order) > max_categories:
             cat_freq = comparison_df[comparison_df['Category'].isin(final_category_order)].groupby('Category')['Proportion'].sum()
             final_category_order = cat_freq.nlargest(max_categories).index.tolist()
             final_category_order = [cat for cat in category_order_override if cat in final_category_order]
             plot_title_suffix = f' (Top {max_categories} of Ordered)'
        else: plot_title_suffix = ' (Ordered)'
    else:
        category_importance = comparison_df.groupby('Category')['Proportion'].mean().sort_values(ascending=False)
        if len(all_categories_present) > max_categories:
            final_category_order = category_importance.nlargest(max_categories).index.tolist()
            plot_title_suffix = f' (Top {max_categories} Overall)'
        else: final_category_order = category_importance.index.tolist()
    comparison_df_filtered = comparison_df[comparison_df['Category'].isin(final_category_order)]
    if comparison_df_filtered.empty: print(f"Skipping comparison plot for '{col_name}': No data remains after filtering."); return
    fig, ax = plt.subplots(figsize=(fixed_width, 7))
    try:
        sns.barplot(x='Category', y='Proportion', hue='DataFrame', data=comparison_df_filtered, order=final_category_order, palette='viridis', ax=ax)
        df_names = list(dataframes.keys()); comp_title = f'Comparison of Normalized "{col_name}"'
        if len(df_names) == 2: comp_title += f' ({df_names[0]} vs {df_names[1]})'
        else: comp_title += f' Across DataFrames'
        ax.set_title(comp_title + plot_title_suffix)
        ax.set_xlabel(col_name); ax.set_ylabel('Proportion'); ax.tick_params(axis='x', rotation=rotation, labelsize='small')
        if rotation != 0: plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
        ax.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
    except Exception as e: print(f"Error plotting comparison barplot for {col_name}: {e}"); plt.close(fig)


# --- Helper function for Top N-grams ---
def get_top_ngrams_list(corpus, ngram_range=(1,1), top_n=20):
    """Calculates and returns the list of top N n-grams."""
    if corpus.empty: return []
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        sum_words = X.sum(axis=0)
        words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()], key=lambda x: x[1], reverse=True)
        top_ngrams = [item[0] for item in words_freq[:top_n]]
        return top_ngrams
    except ValueError as e: print(f"Value error processing n-grams (range {ngram_range}): {e}."); return []
    except Exception as e: print(f"Unexpected error calculating n-grams (range {ngram_range}): {e}"); return []

def plot_top_ngrams(corpus, title, ngram_range=(1,1), top_n=20, figsize=(10, 8), save_path=None):
    """
    Calculates and plots top N n-grams from a text corpus.
    Uses CountVectorizer's default lowercasing and tokenization.
    Optionally saves the plot instead of showing it.
    """
    if corpus.empty: print(f"Skipping n-gram plot '{title}': Empty corpus."); return
    fig, ax = plt.subplots(figsize=figsize)
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(corpus); sum_words = X.sum(axis=0)
        words_freq = sorted([(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()], key=lambda x: x[1], reverse=True)
        if not words_freq: print(f"No {'bigrams' if ngram_range==(2,2) else 'unigrams'} found for '{title}'."); plt.close(fig); return
        top_df = pd.DataFrame(words_freq[:top_n], columns=['Ngram', 'Frequency'])
        sns.barplot(x='Frequency', y='Ngram', data=top_df, palette='viridis', ax=ax)
        ax.set_title(title)
        if save_path:
            try:
                dir_name = os.path.dirname(save_path)
                if dir_name: os.makedirs(dir_name, exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight', dpi=plt.rcParams['figure.dpi'])
                plt.close(fig)
            except Exception as e_save: print(f"Error saving plot to {save_path}: {e_save}"); plt.close(fig)
        else:
            plt.show()
    except ValueError as e: print(f"Error processing n-grams for '{title}': {e}."); plt.close(fig)
    except Exception as e: print(f"An unexpected error occurred during n-gram plotting for '{title}': {e}"); plt.close(fig)


# --- Main EDA Function ---
def comprehensive_nlp_eda(
    dataframes: Dict[str, pd.DataFrame],
    text_col: str ='processed_text',
    target_col: str ='RCC',
    common_meta_discrete: List[str] =['file extension', 'token bucket'],
    common_meta_continuous: List[str] =['number of tokens'],
    specific_meta_discrete: List[str] =['LOB'],
    specific_meta_datetime: List[str] =['FileModifiedTime'],
    oov_reference_df_name: str ='train',
    base_save_path: Optional[str] =None,
    high_dpi: int =150,
    label_rotation: int =45,
    max_categories_plot: int =40,
    plot_width: int =18,
    year_bucket_threshold: int =15,
    year_bucket_size: int =2,
    # Text Analysis Params
    analyze_text_content: bool =True,
    analyze_ngrams: bool =False,
    top_n_terms: int =20,
    analyze_bigrams: bool =True,
    ngram_analysis_sample_size: int =25000,
    ttr_min_samples_per_class: int = 10, # Min samples needed for per-class TTR
    # Specific Comparisons
    specific_comparisons: Optional[List[Tuple[str, str]]] = None
):
    """
    Performs comprehensive EDA for multiclass NLP classification on multiple dataframes.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary mapping dataframe names to DataFrames.
        text_col (str): Name of the text column (assumed preprocessed except for lowercase).
        target_col (str): Name of the target label column.
        common_meta_discrete (List[str]): List of discrete metadata columns common to all dfs.
        common_meta_continuous (List[str]): List of continuous metadata columns common to all dfs.
        specific_meta_discrete (List[str]): List of discrete metadata columns specific to some dfs.
        specific_meta_datetime (List[str]): List of datetime metadata columns specific to some dfs.
        oov_reference_df_name (str): Name of the dataframe in the dictionary to use for reference vocabulary.
        base_save_path (Optional[str]): Base directory path to save specific outputs (like n-grams per class).
        high_dpi (int): DPI setting for matplotlib figures.
        label_rotation (int): Rotation angle for x-axis labels in plots.
        max_categories_plot (int): Maximum number of categories to display in discrete plots.
        plot_width (int): Fixed width for most plots displaying categories.
        year_bucket_threshold (int): Number of unique years above which bucketing is applied.
        year_bucket_size (int): Size of year buckets when applied.
        analyze_text_content (bool): Master toggle for Section 7 (TTR, N-grams).
        analyze_ngrams (bool): Toggle specifically for N-gram plots and overlap calculations (requires analyze_text_content=True).
        top_n_terms (int): Number of top terms/n-grams to display/compare.
        analyze_bigrams (bool): Whether to include bigram analysis (requires analyze_ngrams=True).
        ngram_analysis_sample_size (int): Max documents per dataset sampled for dataset N-gram analysis.
        ttr_min_samples_per_class (int): Min samples needed for per-class TTR calculation/plotting.
        specific_comparisons (Optional[List[Tuple[str, str]]]): List of dataframe name pairs for direct comparison. Example: [('oot', 'prod'), ('train', 'test')].
    """
    # --- Setup ---
    plt.rcParams['figure.dpi'] = high_dpi
    print("="*80); print("Comprehensive NLP EDA Report"); print("="*80)
    target_dfs = {}
    TOKEN_BUCKET_ORDER = ['0-100', '101-500', '501-1000', '1001-5000', '5001-10000', '10000+', '5000+']

    # --- 1. Basic Information ---
    print("\n--- 1. Basic Information ---")
    for name, df in dataframes.items():
        print(f"\n--- DataFrame: {name} ---"); print(f"Shape: {df.shape}")
        print("\nColumns and Data Types:"); df.info()
        print("\nMissing Value Counts:")
        missing_counts = df.isnull().sum(); missing_percent = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({'Count': missing_counts, 'Percentage': missing_percent.round(2)})
        print(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False))
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols: print(f"\nDescriptive Statistics for Numerical Columns ({name}):"); print(df[numerical_cols].describe())
        else: print("\nNo numerical columns found.")
        if target_col in df.columns:
            print(f"\nTarget Variable ('{target_col}') Distribution ({name}):")
            target_value_counts = df[target_col].value_counts(); target_value_counts_norm = df[target_col].value_counts(normalize=True)
            target_dist_df = pd.DataFrame({'Count': target_value_counts, 'Proportion': target_value_counts_norm.round(4)})
            print(target_dist_df);
            if not df[target_col].isnull().all(): target_dfs[name] = df
        else: print(f"\nTarget Variable ('{target_col}') not found in {name}.")

    # --- 2. Metadata Analysis: Discrete Columns ---
    print("\n" + "="*80); print("--- 2. Metadata Analysis: Discrete Columns ---")
    print(f"\n--- Target Column ('{target_col}') Analysis ---")
    if target_dfs:
        for name, df in target_dfs.items(): plot_discrete_distribution(df, target_col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
        compare_discrete_distributions(target_dfs, target_col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
    else: print(f"Target column '{target_col}' not found or all NaN.")
    print("\n--- Common Discrete Metadata Analysis ---")
    for col in common_meta_discrete:
        print(f"\nAnalyzing: {col}")
        order_override = TOKEN_BUCKET_ORDER if col == 'token bucket' else None
        for name, df in dataframes.items():
            if col in df.columns: plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width, category_order_override=order_override)
            else: print(f"Column '{col}' not found in DataFrame '{name}'.")
        compare_discrete_distributions(dataframes, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width, category_order_override=order_override)
    print("\n--- Specific Discrete Metadata Analysis ---")
    for col in specific_meta_discrete:
        print(f"\nAnalyzing: {col}"); specific_dfs = {name: df for name, df in dataframes.items() if col in df.columns}
        if specific_dfs:
            valid_specific_dfs = {n: d for n, d in specific_dfs.items() if not d[col].isnull().all()}
            if valid_specific_dfs:
                order_override = TOKEN_BUCKET_ORDER if col == 'token bucket' else None
                for name, df in valid_specific_dfs.items(): plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width, category_order_override=order_override)
                compare_discrete_distributions(valid_specific_dfs, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width, category_order_override=order_override)
            else: print(f"Column '{col}' found but contains only NaN values.")
        else: print(f"Column '{col}' not found.")

    # --- 3. Metadata Analysis: Continuous Columns ---
    print("\n" + "="*80); print("--- 3. Metadata Analysis: Continuous Columns ---")
    for col in common_meta_continuous:
        print(f"\nAnalyzing: {col}")
        num_dfs_with_col = sum(1 for df in dataframes.values() if col in df.columns and not df[col].isnull().all())
        if num_dfs_with_col > 0: # Histogram
            plt.figure(figsize=(12, 5 * num_dfs_with_col)); plot_index = 1
            for name, df in dataframes.items():
                if col in df.columns and not df[col].isnull().all():
                    plt.subplot(num_dfs_with_col, 1, plot_index); sns.histplot(df[col], kde=True, bins=50)
                    plt.title(f'{name}: Distribution of {col}'); plt.xlabel(col); plt.ylabel('Frequency'); plot_index += 1
                elif col in df.columns and df[col].isnull().all(): print(f"Column '{col}' in '{name}' NaN.")
                else: print(f"Column '{col}' not found in '{name}'.")
            if plot_index > 1: plt.suptitle(f'Histograms/KDE for: {col}', fontsize=16, y=1.0); plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()
            else: plt.close()
        else: print(f"Column '{col}' not found or all NaN.")
        plot_data_boxplot = [] # Comparison Boxplot
        for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 temp_df = df[[col]].dropna().copy();
                 if not temp_df.empty: temp_df['DataFrame'] = name; plot_data_boxplot.append(temp_df)
        if plot_data_boxplot:
             fig, ax = plt.subplots(figsize=(10, 6)); combined_df_boxplot = pd.concat(plot_data_boxplot, ignore_index=True)
             sns.boxplot(x='DataFrame', y=col, data=combined_df_boxplot, palette='viridis', showfliers=False, ax=ax)
             ax.set_title(f'Comparison of "{col}" Distribution (Outliers Hidden)'); ax.set_xlabel('DataFrame'); ax.set_ylabel(col); plt.show()
        print(f"\n--- Distribution of '{col}' by Target ('{target_col}') ---") # Target Boxplot
        target_dfs_with_col = {name: df for name, df in target_dfs.items() if col in df.columns and not df[col].isnull().all()}
        if target_dfs_with_col:
            for name, df in target_dfs_with_col.items():
                if not df[target_col].isnull().all():
                    df_plot = df[[col, target_col]].dropna().copy()
                    if df_plot.empty: print(f"Skipping box plot for {name}."); continue
                    df_plot[target_col] = df_plot[target_col].astype(str); target_order = sorted(df_plot[target_col].unique())
                    fig, ax = plt.subplots(figsize=(plot_width, 7));
                    sns.boxplot(x=target_col, y=col, data=df_plot, palette='viridis', order=target_order, showfliers=False, ax=ax)
                    ax.set_title(f'{name}: Dist of "{col}" by "{target_col}" (Outliers Hidden)'); ax.set_xlabel(target_col); ax.set_ylabel(col);
                    ax.tick_params(axis='x', rotation=label_rotation)
                    is_positive = (df_plot[col] > 0) if pd.api.types.is_numeric_dtype(df_plot[col]) else pd.Series(False, index=df_plot.index)
                    if is_positive.all() and col == 'number of tokens' and (df_plot[col].max() / df_plot[col].median() > 50):
                        ax.set_yscale('log'); ax.set_ylabel(f"{col} (Log Scale)"); print(f"Applied log scale for {name}.")
                    elif col == 'number of tokens' and not is_positive.all() and is_positive.any(): print(f"Note: Log scale not applied for {name} (non-positive values).")
                    fig.tight_layout(); plt.show()
                else: print(f"Target column in '{name}' NaN.")
        else: print(f"Could not perform analysis by target for '{col}'.")

    # --- 4. Metadata Analysis: Datetime Columns ---
    print("\n" + "="*80); print("--- 4. Metadata Analysis: Datetime Columns ---")
    for col in specific_meta_datetime:
         print(f"\nAnalyzing: {col}"); dt_dfs = {}
         for name, df in dataframes.items(): # Conversion unchanged
             if col in df.columns and not df[col].isnull().all():
                 if pd.api.types.is_datetime64_any_dtype(df[col]): dt_dfs[name] = df.copy()
                 else:
                     print(f"Attempting datetime conversion for '{col}' in '{name}'...")
                     try:
                         df_copy = df.copy(); df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                         if not df_copy[col].isnull().all(): dt_dfs[name] = df_copy; print(f"Conversion successful.")
                         else: print(f"Conversion resulted in all NaNs. Skipping.")
                     except Exception as e: print(f"Could not convert: {e}. Skipping.")
             elif col in df.columns and df[col].isnull().all(): print(f"Column '{col}' in '{name}' NaN.")
         if dt_dfs: # Bucketing/plotting unchanged
             for name, df_dt in dt_dfs.items():
                  print(f"\n--- Datetime Analysis for {col} in {name} ---"); df_dt_nonan = df_dt.dropna(subset=[col]).copy()
                  if df_dt_nonan.empty: print(f"No valid datetime values."); continue
                  df_dt_nonan['Year'] = df_dt_nonan[col].dt.year; yearly_counts = df_dt_nonan['Year'].value_counts().sort_index(); unique_years = yearly_counts.index.astype(int)
                  if not yearly_counts.empty:
                      num_years = len(unique_years); plot_title = f'{name}: Document Count'; x_label = 'Year'
                      if num_years > year_bucket_threshold: # Bucketing unchanged
                          print(f"Bucketing years ({num_years} > {year_bucket_threshold})."); min_year, max_year = unique_years.min(), unique_years.max(); actual_bucket_size = max(1, year_bucket_size)
                          bins = list(range(min_year, max_year + actual_bucket_size, actual_bucket_size));
                          if len(bins)>1 and bins[-1] <= max_year : bins.append(max_year + 1)
                          elif len(bins)==1: bins.append(max_year + 1)
                          labels = [f"{bins[i]}-{bins[i+1]-1}" if bins[i+1]-1 > bins[i] else f"{bins[i]}" for i in range(len(bins)-1)]
                          df_dt_nonan['Year Bucket'] = pd.cut(df_dt_nonan['Year'], bins=bins, labels=labels, right=False, include_lowest=True)
                          counts_to_plot = df_dt_nonan['Year Bucket'].value_counts().sort_index(); plot_title += f' per {actual_bucket_size}-Year Bucket'; x_label = f'{actual_bucket_size}-Year Bucket'
                      else: counts_to_plot = yearly_counts; plot_title += ' per Year'
                      fig, ax = plt.subplots(figsize=(plot_width, 6)); counts_to_plot.plot(kind='bar', color=sns.color_palette('viridis', len(counts_to_plot)), ax=ax)
                      ax.set_title(plot_title + f' ({col})'); ax.set_xlabel(x_label); ax.set_ylabel('Number of Documents');
                      ax.tick_params(axis='x', rotation=label_rotation)
                      ax.grid(True, axis='y'); fig.tight_layout(); plt.show()
                  else: print(f"No non-NaN data points for yearly counts.")
         else: print(f"Column '{col}' not found or unusable.")

    # --- 5. Out-of-Vocabulary (OOV) Analysis ---
    print("\n" + "="*80); print("--- 5. Out-of-Vocabulary (OOV) Analysis ---")
    if oov_reference_df_name not in dataframes: print(f"Error: Reference DF '{oov_reference_df_name}' not found.")
    else:
        ref_df = dataframes[oov_reference_df_name]
        if text_col not in ref_df.columns: print(f"Error: Text column '{text_col}' not found.")
        elif ref_df[text_col].isnull().all(): print(f"Error: Text column '{text_col}' NaN.")
        else:
            print(f"Building vocab from '{oov_reference_df_name}' ('{text_col}')..."); vocab_counter = Counter(); ref_vocab_set = set(); total_ref_tokens = 0
            for text in tqdm(ref_df[text_col].dropna(), desc=f"Building Vocab ({oov_reference_df_name})"): tokens = basic_tokenizer(text); vocab_counter.update(tokens); ref_vocab_set.update(tokens); total_ref_tokens += len(tokens)
            vocab_set = ref_vocab_set; print(f"Vocab size: {len(vocab_set)} unique."); print(f"Total ref tokens: {total_ref_tokens}"); print("\nCalculating OOV %:")
            oov_results = {}; unique_oov_results = {}
            for name, df in dataframes.items():
                if name == oov_reference_df_name: continue
                if text_col not in df.columns or df[text_col].isnull().all(): print(f"Warning: Skipping OOV for '{name}'."); continue
                oov_count = 0; total_tokens = 0; oov_word_set = set(); target_word_set = set()
                for text in tqdm(df[text_col].dropna(), desc=f"Calculating OOV ({name})"):
                    tokens = basic_tokenizer(text); target_word_set.update(tokens)
                    current_oov_tokens = [token for token in tokens if token not in vocab_set]
                    oov_count += len(current_oov_tokens); oov_word_set.update(current_oov_tokens); total_tokens += len(tokens)
                oov_results[name] = (oov_count / total_tokens) * 100 if total_tokens > 0 else np.nan
                unique_oov_results[name] = (len(oov_word_set) / len(target_word_set)) * 100 if len(target_word_set) > 0 else np.nan
                print(f"- {name}:")
                if pd.notna(oov_results.get(name)): print(f"  - OOV % (Token): {oov_results[name]:.2f}%")
                else: print(f"  - No text for token OOV.")
                if pd.notna(unique_oov_results.get(name)): print(f"  - OOV % (Unique): {unique_oov_results[name]:.2f}%")
                else: print(f"  - No text for unique OOV.")
            valid_oov = {k: v for k, v in oov_results.items() if pd.notna(v)}; valid_unique_oov = {k: v for k, v in unique_oov_results.items() if pd.notna(v)}
            if valid_oov:
                fig, ax = plt.subplots(figsize=(max(6, len(valid_oov)*1.5), 5)); s = pd.Series(valid_oov).sort_values(); sns.barplot(x=s.index, y=s.values, palette='viridis', ax=ax); ax.set_title(f'OOV % (Token vs. "{oov_reference_df_name}")'); ax.set_ylabel('OOV %');
                plt.xticks(rotation=label_rotation); fig.tight_layout(); plt.show()
            if valid_unique_oov:
                 fig, ax = plt.subplots(figsize=(max(6, len(valid_unique_oov)*1.5), 5)); s = pd.Series(valid_unique_oov).sort_values(); sns.barplot(x=s.index, y=s.values, palette='magma', ax=ax); ax.set_title(f'OOV % (Unique Word vs. "{oov_reference_df_name}")'); ax.set_ylabel('OOV %');
                 plt.xticks(rotation=label_rotation); fig.tight_layout(); plt.show()

    # --- 6. Cross-Feature Analysis (Examples) ---
    print("\n" + "="*80); print("--- 6. Cross-Feature Analysis (Examples) ---")
    # ... (Code unchanged) ...
    col1_ex1 = common_meta_discrete[0] if common_meta_discrete else None; col2_ex1 = common_meta_continuous[0] if common_meta_continuous else None
    if col1_ex1 and col2_ex1: # Example 1
        print(f"\n--- Analyzing: '{col1_ex1}' vs '{col2_ex1}' ---"); data = []
        for name, df in dataframes.items():
            if col1_ex1 in df.columns and col2_ex1 in df.columns:
                 if not df[col1_ex1].isnull().all() and not df[col2_ex1].isnull().all():
                     tdf = df[[col1_ex1, col2_ex1]].dropna().copy()
                     if not tdf.empty: tdf['DataFrame'] = name; tdf[col1_ex1] = tdf[col1_ex1].astype(str); data.append(tdf)
        if data:
            cdf = pd.concat(data, ignore_index=True)
            order_ex1 = TOKEN_BUCKET_ORDER if col1_ex1 == 'token bucket' else cdf[col1_ex1].value_counts().index[:max_categories_plot].tolist()
            order_ex1_filtered = [cat for cat in order_ex1 if cat in cdf[col1_ex1].unique()]
            fig, ax = plt.subplots(figsize=(plot_width, 7)); sns.boxplot(x=col1_ex1, y=col2_ex1, hue='DataFrame', data=cdf, palette='viridis', order=order_ex1_filtered, showfliers=False, ax=ax)
            ax.set_title(f'Relationship: "{col1_ex1}" vs "{col2_ex1}" (Outliers Hidden)'); ax.set_xlabel(col1_ex1); ax.set_ylabel(col2_ex1);
            ax.tick_params(axis='x', rotation=label_rotation)
            ax.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
        else: print(f"Not enough data.")
    else: print("\nSkipping Cross-Feature Ex1.")
    col1_ex2 = common_meta_discrete[0] if common_meta_discrete else None
    if col1_ex2 and target_col: # Example 2
        print(f"\n--- Analyzing: '{col1_ex2}' vs '{target_col}' ---")
        dfs = {name: df for name, df in target_dfs.items() if col1_ex2 in df.columns and not df[col1_ex2].isnull().all() and not df[target_col].isnull().all()}
        if dfs:
            for name, df in dfs.items():
                 dfp = df[[col1_ex2, target_col]].dropna().copy();
                 if dfp.empty: continue
                 dfp[col1_ex2] = dfp[col1_ex2].astype(str); dfp[target_col] = dfp[target_col].astype(str)
                 try:
                    ct = pd.crosstab(dfp[col1_ex2], dfp[target_col], normalize='index') * 100
                    if col1_ex2 == 'token bucket': order_ex2 = [b for b in TOKEN_BUCKET_ORDER if b in ct.index]
                    else: order_ex2 = dfp[col1_ex2].value_counts().index[:max_categories_plot].tolist()
                    ct = ct.reindex(order_ex2).dropna(how='all')
                    if ct.empty: continue
                    fig, ax = plt.subplots(figsize=(plot_width, 7)); ct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax); ax.set_title(f'{name}: Proportion of "{target_col}" within "{col1_ex2}" (Top {len(order_ex2)} Cats)'); ax.set_xlabel(col1_ex2); ax.set_ylabel('%');
                    ax.tick_params(axis='x', rotation=label_rotation)
                    ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
                 except Exception as e: print(f"Error plotting stacked bar for {name}: {e}")
        else: print(f"Not enough data.")
    elif not col1_ex2: print("\nSkipping Cross-Feature Ex2.")


    # --- 7. Text Content Analysis ---
    if analyze_text_content:
        print("\n" + "="*80); print("--- 7. Text Content Analysis ---")

        # --- 7a. Type-Token Ratio (TTR) ---
        print(f"\n--- 7a. Type-Token Ratio (Lexical Diversity, Stopwords Removed) ---")
        ttr_data_list = [] # Store results for final plot
        # Calculate TTR per Dataset (no change in logic, just storing)
        print("\nCalculating TTR per Dataset:")
        for name, df in dataframes.items():
            if text_col not in df.columns or df[text_col].isnull().all(): print(f"- {name}: Skipping"); continue
            token_lists = df[text_col].dropna().apply(lambda x: [token for token in basic_tokenizer(x) if token not in STOPWORDS])
            all_tokens_filtered = [token for sublist in token_lists for token in sublist]
            if not all_tokens_filtered: print(f"- {name}: Skipping (No non-stopword tokens)"); continue
            total_tokens_filtered = len(all_tokens_filtered); unique_tokens_filtered = len(set(all_tokens_filtered))
            ttr = (unique_tokens_filtered / total_tokens_filtered) * 100 if total_tokens_filtered > 0 else 0
            print(f"- {name}: Unique={unique_tokens_filtered}, Total={total_tokens_filtered}, TTR={ttr:.2f}%")
            # Add dataset level TTR for potential combined plot later if needed
            # ttr_data_list.append({'Dataset': name, 'RCC': 'Overall', 'TTR': ttr})

        # Calculate TTR per Class for Train, Test, OOT
        print(f"\nCalculating TTR per Class (for Train, Test, OOT if available):")
        all_classes_found = set()
        for dataset_name in [name for name in [oov_reference_df_name, 'test', 'oot'] if name in dataframes]:
             df_ttr = dataframes.get(dataset_name)
             if df_ttr is None or target_col not in df_ttr.columns or text_col not in df_ttr.columns:
                 print(f"Skipping TTR per class for '{dataset_name}': Missing DF or required columns.")
                 continue
             print(f"Processing '{dataset_name}' for per-class TTR...")
             grouped = df_ttr.dropna(subset=[text_col, target_col]).groupby(target_col)
             for class_label, group_df in grouped:
                 if len(group_df) < ttr_min_samples_per_class:
                      print(f"  - Skipping Class '{class_label}' in '{dataset_name}': {len(group_df)} samples < {ttr_min_samples_per_class}")
                      continue # Skip if too few samples
                 all_classes_found.add(class_label) # Track all classes we calculate for
                 token_lists = group_df[text_col].dropna().apply(lambda x: [token for token in basic_tokenizer(x) if token not in STOPWORDS])
                 all_tokens_filtered = [token for sublist in token_lists for token in sublist]
                 if not all_tokens_filtered: continue # Skip if no tokens after filtering
                 total_tokens_filtered = len(all_tokens_filtered); unique_tokens_filtered = len(set(all_tokens_filtered))
                 ttr = (unique_tokens_filtered / total_tokens_filtered) * 100 if total_tokens_filtered > 0 else 0
                 ttr_data_list.append({'Dataset': dataset_name, 'RCC': str(class_label), 'TTR': ttr})
                 # No individual print here

        # Plot TTR Comparison per Class
        if ttr_data_list:
            ttr_df_plot = pd.DataFrame(ttr_data_list)
            # Order classes alphabetically for consistent plotting
            class_order = sorted(list(all_classes_found), key=str)
            plt.figure(figsize=(max(10, len(class_order)*1.5), 6)) # Adjust width based on num classes
            sns.barplot(data=ttr_df_plot, x='RCC', y='TTR', hue='Dataset', order=class_order, palette='viridis')
            plt.ylabel("TTR (%) (Stopwords Removed)"); plt.xlabel("RCC Class")
            plt.title(f"TTR per Class Comparison (Min {ttr_min_samples_per_class} Samples per Class/Dataset)")
            plt.xticks(rotation=label_rotation, ha='right' if label_rotation else 'center')
            plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left');
            plt.tight_layout(rect=[0, 0, 0.9, 1]);
            plt.show()
        else:
            print("\nNo sufficient data found to generate TTR per class comparison plot.")


        # --- N-Gram Analysis (Conditional) ---
        if analyze_ngrams:
            print("\n--- N-Gram Analysis Enabled ---")
            # --- 7b. Top N-grams per Class (Save plots) ---
            print(f"\n--- 7b. Top N-grams per Class (Target Column: '{target_col}') ---")
            top_ngrams_data = {} # Structure: {dataset_name: {class_label: {'unigrams': {set}, 'bigrams': {set}}}}
            datasets_to_process_class = [name for name, df in dataframes.items() if name in [oov_reference_df_name, 'test', 'oot'] and target_col in df.columns and text_col in df.columns and not df[text_col].isnull().all()]
            print(f"Processing datasets: {', '.join(datasets_to_process_class)} for per-class N-grams...")

            for dataset_name in datasets_to_process_class:
                 df_ngram = dataframes.get(dataset_name); top_ngrams_data[dataset_name] = {}
                 grouped = df_ngram.dropna(subset=[text_col, target_col]).groupby(target_col)
                 print(f"\n-> Analyzing Top {top_n_terms} Unigrams for '{dataset_name}' (saving plots if path provided)...")
                 for class_label, group_df in tqdm(grouped, desc=f"Unigrams per Class ({dataset_name})"):
                     corpus = group_df[text_col].dropna().astype(str)
                     if not corpus.empty:
                         top_unigrams_list = get_top_ngrams_list(corpus, ngram_range=(1,1), top_n=top_n_terms)
                         if class_label not in top_ngrams_data[dataset_name]: top_ngrams_data[dataset_name][class_label] = {}
                         top_ngrams_data[dataset_name][class_label]['unigrams'] = set(top_unigrams_list)
                         save_filepath = None
                         if base_save_path: safe_class_label = re.sub(r'[^\w\-]+', '_', str(class_label)); save_dir = os.path.join(base_save_path, "ngram_analysis", "ngrams_per_class", dataset_name, safe_class_label); save_filename = f"top_{top_n_terms}_unigrams.png"; save_filepath = os.path.join(save_dir, save_filename); plot_top_ngrams(corpus, title=f"Top {top_n_terms} Unigrams: Class {class_label} ({dataset_name})", ngram_range=(1,1), top_n=top_n_terms, save_path=save_filepath); # if save_filepath and os.path.exists(save_filepath): print(f"Saved: {save_filepath}", end='\r')
                 print() # Newline
                 if analyze_bigrams:
                     print(f"\n-> Analyzing Top {top_n_terms} Bigrams for '{dataset_name}' (saving plots if path provided)...")
                     for class_label, group_df in tqdm(grouped, desc=f"Bigrams per Class ({dataset_name})"):
                          corpus = group_df[text_col].dropna().astype(str)
                          if not corpus.empty:
                              top_bigrams_list = get_top_ngrams_list(corpus, ngram_range=(2,2), top_n=top_n_terms)
                              if class_label not in top_ngrams_data[dataset_name]: top_ngrams_data[dataset_name][class_label] = {}
                              top_ngrams_data[dataset_name][class_label]['bigrams'] = set(top_bigrams_list)
                              save_filepath = None
                              if base_save_path: safe_class_label = re.sub(r'[^\w\-]+', '_', str(class_label)); save_dir = os.path.join(base_save_path, "ngram_analysis", "ngrams_per_class", dataset_name, safe_class_label); save_filename = f"top_{top_n_terms}_bigrams.png"; save_filepath = os.path.join(save_dir, save_filename); plot_top_ngrams(corpus, title=f"Top {top_n_terms} Bigrams: Class {class_label} ({dataset_name})", ngram_range=(2,2), top_n=top_n_terms, save_path=save_filepath); # if save_filepath and os.path.exists(save_filepath): print(f"Saved: {save_filepath}", end='\r')
                     print() # Newline

            # --- 7c. Class N-gram Overlap DataFrame ---
            print(f"\n--- 7c. Class N-gram Overlap vs '{oov_reference_df_name}' ---")
            overlap_results = []
            ref_ngrams_per_class = top_ngrams_data.get(oov_reference_df_name, {})
            if not ref_ngrams_per_class: print(f"Cannot calculate class overlap: No N-gram data for reference '{oov_reference_df_name}'.")
            else:
                datasets_to_compare = [ds for ds in ['test', 'oot'] if ds in top_ngrams_data]
                for compare_ds_name in datasets_to_compare:
                    compare_ngrams_per_class = top_ngrams_data.get(compare_ds_name, {})
                    if not compare_ngrams_per_class: print(f"Skipping overlap for '{compare_ds_name}'."); continue
                    print(f"\nComparing '{compare_ds_name}' classes to '{oov_reference_df_name}' classes:")
                    all_classes = sorted(list(set(ref_ngrams_per_class.keys()) | set(compare_ngrams_per_class.keys())))
                    for class_label in all_classes:
                        ref_class_data = ref_ngrams_per_class.get(class_label, {}); compare_class_data = compare_ngrams_per_class.get(class_label, {})
                        ref_uni = ref_class_data.get('unigrams', set()); comp_uni = compare_class_data.get('unigrams', set())
                        uni_overlap = len(ref_uni & comp_uni); uni_overlap_pct = (uni_overlap / len(ref_uni)) * 100 if ref_uni else 0
                        result_row = {'Comparison': f"{oov_reference_df_name} vs {compare_ds_name}", 'RCC': class_label, f'Unigram Overlap (%)': f"{uni_overlap_pct:.1f}%"}
                        if analyze_bigrams:
                            ref_bi = ref_class_data.get('bigrams', set()); comp_bi = compare_class_data.get('bigrams', set())
                            bi_overlap = len(ref_bi & comp_bi); bi_overlap_pct = (bi_overlap / len(ref_bi)) * 100 if ref_bi else 0
                            result_row[f'Bigram Overlap (%)'] = f"{bi_overlap_pct:.1f}%"
                        overlap_results.append(result_row)
                if overlap_results:
                    overlap_df = pd.DataFrame(overlap_results); cols_order = ['Comparison', 'RCC', f'Unigram Overlap (%)'];
                    if analyze_bigrams: cols_order.append(f'Bigram Overlap (%)')
                    print(overlap_df[cols_order].to_string())
                else: print("No overlap results to display.")


            # --- 7d. Top N-grams per Dataset (Overall Overlap) ---
            print(f"\n--- 7d. Top N-grams per Dataset & Overall Overlap with {oov_reference_df_name} ---")
            ref_top_unigrams_overall = set(get_top_ngrams_list(dataframes[oov_reference_df_name][text_col].dropna().astype(str), ngram_range=(1,1), top_n=top_n_terms)) if oov_reference_df_name in dataframes else set()
            ref_top_bigrams_overall = set(get_top_ngrams_list(dataframes[oov_reference_df_name][text_col].dropna().astype(str), ngram_range=(2,2), top_n=top_n_terms)) if oov_reference_df_name in dataframes and analyze_bigrams else set()

            print(f"\nAnalyzing top {top_n_terms} Unigrams per Dataset (sampling large DFs to max {ngram_analysis_sample_size})...")
            for name, df in dataframes.items():
                 if text_col not in df.columns or df[text_col].isnull().all(): print(f"- {name}: Skipping Unigrams."); continue
                 df_sampled = df;
                 if len(df) > ngram_analysis_sample_size: print(f"- {name}: Sampling (size {len(df)}) to {ngram_analysis_sample_size}."); df_sampled = df.sample(n=ngram_analysis_sample_size, random_state=42)
                 corpus = df_sampled[text_col].dropna().astype(str)
                 if not corpus.empty:
                     plot_top_ngrams(corpus, title=f"Top {top_n_terms} Unigrams for Dataset: {name}", ngram_range=(1,1), top_n=top_n_terms, save_path=None)
                     if name != oov_reference_df_name and ref_top_unigrams_overall:
                         current_top_unigrams = set(get_top_ngrams_list(corpus, ngram_range=(1,1), top_n=top_n_terms))
                         overlap_count = len(current_top_unigrams & ref_top_unigrams_overall)
                         denominator = len(ref_top_unigrams_overall)
                         overlap_percent = (overlap_count / denominator) * 100 if denominator > 0 else 0
                         print(f"  - Overall Unigram Overlap with '{oov_reference_df_name}' Top {top_n_terms}: {overlap_count}/{denominator} ({overlap_percent:.1f}%)")
                 else: print(f"- {name}: Skipping Unigrams (No text data).")

            if analyze_bigrams:
                print(f"\nAnalyzing top {top_n_terms} Bigrams per Dataset (sampling large DFs to max {ngram_analysis_sample_size})...")
                for name, df in dataframes.items():
                     if text_col not in df.columns or df[text_col].isnull().all(): print(f"- {name}: Skipping Bigrams."); continue
                     df_sampled = df;
                     if len(df) > ngram_analysis_sample_size: df_sampled = df.sample(n=ngram_analysis_sample_size, random_state=42)
                     corpus = df_sampled[text_col].dropna().astype(str)
                     if not corpus.empty:
                         plot_top_ngrams(corpus, title=f"Top {top_n_terms} Bigrams for Dataset: {name}", ngram_range=(2,2), top_n=top_n_terms, save_path=None)
                         if name != oov_reference_df_name and ref_top_bigrams_overall:
                             current_top_bigrams = set(get_top_ngrams_list(corpus, ngram_range=(2,2), top_n=top_n_terms))
                             overlap_count = len(current_top_bigrams & ref_top_bigrams_overall)
                             denominator = len(ref_top_bigrams_overall)
                             overlap_percent = (overlap_count / denominator) * 100 if denominator > 0 else 0
                             print(f"  - Overall Bigram Overlap with '{oov_reference_df_name}' Top {top_n_terms}: {overlap_count}/{denominator} ({overlap_percent:.1f}%)")
                     else: print(f"- {name}: Skipping Bigrams (No text data).")
        else:
             print("\nSkipping N-Gram Analysis (analyze_ngrams=False).")

    else:
        print("\nSkipping Text Content Analysis (analyze_text_content=False).")


    # --- Section 8: Specific Dataset Comparisons (Modified Datetime Plot) ---
    if specific_comparisons:
        print("\n" + "="*80); print("--- 8. Specific Dataset Comparisons ---")
        if not isinstance(specific_comparisons, (list, tuple)):
            print("Warning: 'specific_comparisons' should be a list of tuples. Skipping.")
        else:
            for comp_pair in specific_comparisons:
                if not isinstance(comp_pair, (list, tuple)) or len(comp_pair) != 2:
                     print(f"Warning: Invalid item in 'specific_comparisons': {comp_pair}. Skipping.")
                     continue

                name1, name2 = comp_pair
                print(f"\n--- Comparing: '{name1}' vs '{name2}' ---")

                if name1 not in dataframes or name2 not in dataframes:
                     print(f"Error: One or both dataframes ('{name1}', '{name2}') not found. Skipping comparison.")
                     continue

                df1 = dataframes[name1]; df2 = dataframes[name2]
                common_cols = list(set(df1.columns) & set(df2.columns) - {text_col, target_col})
                if not common_cols: print(f"No common metadata columns found between '{name1}' and '{name2}'."); continue

                print(f"Common columns found: {', '.join(common_cols)}")
                comparison_dfs_dict = {name1: df1, name2: df2}

                all_discrete_defs = common_meta_discrete + specific_meta_discrete; all_continuous_defs = common_meta_continuous; all_datetime_defs = specific_meta_datetime
                discrete_common = [col for col in common_cols if col in all_discrete_defs]
                continuous_common = [col for col in common_cols if col in all_continuous_defs]
                datetime_common = [col for col in common_cols if col in all_datetime_defs]
                other_common = [col for col in common_cols if col not in discrete_common + continuous_common + datetime_common]
                if other_common:
                     print(f"Attempting to classify remaining common columns: {other_common}")
                     for col in other_common:
                         try:
                             if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]): continuous_common.append(col); print(f" -> Classified '{col}' as Continuous")
                             elif pd.api.types.is_object_dtype(df1[col]) or pd.api.types.is_categorical_dtype(df1[col]): discrete_common.append(col); print(f" -> Classified '{col}' as Discrete")
                             elif pd.api.types.is_datetime64_any_dtype(df1[col]) and pd.api.types.is_datetime64_any_dtype(df2[col]): datetime_common.append(col); print(f" -> Classified '{col}' as Datetime")
                             else: print(f" -> Could not auto-classify '{col}'. Skipping.")
                         except Exception as e_classify: print(f" -> Error classifying '{col}': {e_classify}. Skipping.")

                # Compare Discrete Columns (uses compare_discrete_distributions which plots proportions)
                if discrete_common:
                     print(f"\nComparing Discrete Columns (Proportions): {', '.join(discrete_common)}")
                     for col in discrete_common:
                         order_override = TOKEN_BUCKET_ORDER if col == 'token bucket' else None
                         compare_discrete_distributions(comparison_dfs_dict, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width, category_order_override=order_override)

                # Compare Continuous Columns (Boxplot comparing distributions)
                if continuous_common:
                     print(f"\nComparing Continuous Columns: {', '.join(continuous_common)}")
                     for col in continuous_common:
                         plot_data_comp = []
                         for name, df in comparison_dfs_dict.items():
                             if col in df.columns and not df[col].isnull().all():
                                 temp_df = df[[col]].dropna().copy()
                                 if not temp_df.empty: temp_df['DataFrame'] = name; plot_data_comp.append(temp_df)
                         if len(plot_data_comp) == 2:
                             fig, ax = plt.subplots(figsize=(8, 6)); combined_df_comp = pd.concat(plot_data_comp, ignore_index=True)
                             sns.boxplot(x='DataFrame', y=col, data=combined_df_comp, palette='viridis', showfliers=False, ax=ax)
                             ax.set_title(f'Comparison of "{col}" ({name1} vs {name2}, Outliers Hidden)'); ax.set_xlabel(''); ax.set_ylabel(col); plt.show()
                         else: print(f"Skipping comparison for '{col}': Not enough valid data.")

                # Compare Datetime Columns (Normalized Proportions)
                if datetime_common:
                     print(f"\nComparing Datetime Columns (Proportions): {', '.join(datetime_common)}")
                     for col in datetime_common:
                          print(f"\n--- Analyzing Datetime Proportions: {col} for {name1} vs {name2} ---")
                          dt_dfs_comp = {}; valid_dt_data = []
                          # Validate/Convert Datetime column for the pair
                          for name, df in comparison_dfs_dict.items():
                             df_copy = None
                             if col in df.columns and not df[col].isnull().all():
                                 if pd.api.types.is_datetime64_any_dtype(df[col]): df_copy = df.copy()
                                 else:
                                     print(f"Attempting dt conversion for '{col}' in '{name}'...");
                                     try: temp_df = df.copy(); temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce');
                                         if not temp_df[col].isnull().all(): df_copy = temp_df; print(f" -> Success.")
                                         else: print(f" -> Failed (all NaNs).")
                                     except Exception as e: print(f" -> Failed: {e}.")
                             if df_copy is not None: dt_dfs_comp[name] = df_copy; valid_dt_data.append(df_copy.dropna(subset=[col])[[col]])
                          if len(dt_dfs_comp) != 2: print(f"Skipping comparison for '{col}'."); continue
                          if not valid_dt_data: print("No valid datetime values."); continue
                          combined_dates = pd.concat(valid_dt_data)[col]
                          if combined_dates.empty: print("No valid datetime values."); continue

                          # Determine common buckets based on combined range
                          min_date = combined_dates.min(); max_date = combined_dates.max(); min_year = min_date.year; max_year = max_date.year
                          unique_years_overall = list(range(min_year, max_year + 1)); num_years = len(unique_years_overall)
                          bucket_col_name = f"{col}_Bucket"; use_buckets = num_years > year_bucket_threshold
                          bins = None; labels = None; x_axis_label = "Year"; plot_title_suffix = f" ({name1} vs {name2})"
                          if use_buckets:
                              print(f"Bucketing years ({num_years} > {year_bucket_threshold})."); actual_bucket_size = max(1, year_bucket_size)
                              bins = list(range(min_year, max_year + actual_bucket_size, actual_bucket_size));
                              if len(bins)>1 and bins[-1] <= max_year : bins.append(max_year + 1)
                              elif len(bins)==1: bins.append(max_year + 1)
                              labels = [f"{bins[i]}-{bins[i+1]-1}" if bins[i+1]-1 > bins[i] else f"{bins[i]}" for i in range(len(bins)-1)]
                              x_axis_label = f'{actual_bucket_size}-Year Bucket'; plot_title_suffix += f' per {x_axis_label}'
                          else: labels = sorted([str(y) for y in unique_years_overall]); plot_title_suffix += ' per Year'

                          # --- Calculate Normalized Counts per DF ---
                          plot_data_dt_norm = []
                          all_bucket_labels_ordered = labels
                          for name, df_dt in dt_dfs_comp.items():
                              df_dt_nonan = df_dt.dropna(subset=[col]).copy()
                              total_valid_docs = len(df_dt_nonan) # Denominator for normalization

                              if total_valid_docs == 0: # Add zero proportions if df has no data
                                   proportions = pd.Series(0.0, index=pd.CategoricalIndex(all_bucket_labels_ordered, ordered=True))
                              else:
                                  df_dt_nonan['Year'] = df_dt_nonan[col].dt.year
                                  if use_buckets: df_dt_nonan[bucket_col_name] = pd.cut(df_dt_nonan['Year'], bins=bins, labels=labels, right=False, include_lowest=True)
                                  else: df_dt_nonan[bucket_col_name] = df_dt_nonan['Year'].astype(str)
                                  df_dt_nonan[bucket_col_name] = pd.Categorical(df_dt_nonan[bucket_col_name], categories=all_bucket_labels_ordered, ordered=True)
                                  # Calculate proportions instead of counts
                                  proportions = df_dt_nonan[bucket_col_name].value_counts(normalize=True) # Use normalize=True

                              # Reindex counts to include all possible buckets/labels with 0 prop if missing
                              proportions = proportions.reindex(all_bucket_labels_ordered, fill_value=0.0)
                              df_props = pd.DataFrame({'Proportion': proportions})
                              df_props['DataFrame'] = name
                              df_props['Bucket'] = proportions.index
                              plot_data_dt_norm.append(df_props)

                          # Plot combined proportions
                          if not plot_data_dt_norm: print("No data to plot for datetime proportion comparison."); continue
                          combined_dt_props = pd.concat(plot_data_dt_norm).reset_index(drop=True)

                          fig, ax = plt.subplots(figsize=(plot_width, 6))
                          sns.barplot(x='Bucket', y='Proportion', hue='DataFrame', data=combined_dt_props, palette='viridis', ax=ax) # Plot Proportion
                          ax.set_title(f'Document Proportion Comparison for "{col}"{plot_title_suffix}')
                          ax.set_xlabel(x_axis_label); ax.set_ylabel('Proportion of Documents'); # Update Y label
                          ax.tick_params(axis='x', rotation=label_rotation, labelsize='small')
                          if label_rotation != 0: plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
                          ax.grid(True, axis='y');
                          ax.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left'); # Ensure legend is shown
                          fig.tight_layout(rect=[0, 0, 0.9, 1]); # Adjust layout for legend
                          plt.show()
    else:
        print("\nNo specific dataset comparisons requested.")

    print("\n" + "="*80); print("EDA Complete."); print("="*80)

# --- Example Function Call ---
# (Example call structure remains the same)

# Create placeholder DataFrames for the example call to run without error
# --- REPLACE THIS WITH YOUR ACTUAL DATAFRAMES ---
placeholder_data = {'processed_text': ['Text A Train Document about apples', 'TEXT B ABOUT MODELS and oranges', 'Train specific Words here apples oranges the a is'], 'file extension': ['.pdf', '.docx', '.pdf'], 'number of tokens': [100, 200, 150], 'token bucket': ['100-500', '100-500', '100-500'], 'RCC': ['ClassA', 'ClassB', 'ClassA'], 'FileModifiedTime': pd.to_datetime(['2022-01-10', '2022-05-15', '2023-03-20']), 'LOB': ['Finance', 'HR', 'Finance']}
placeholder_data_oot = {'processed_text': ['Text c oot version with apples', 'OOT unique content no fruit the'], 'file extension': ['.txt', '.txt'], 'number of tokens': [50, 60], 'token bucket': ['0-100', '0-100'], 'RCC': ['ClassA', 'ClassC'], 'FileModifiedTime': pd.to_datetime(['2023-01-15', '2023-02-20']), 'LOB': ['Finance', 'Legal']}
prod_texts = [f'Prod text {i} example Content for Production apples' for i in range(100)] + [f'Prod text {i} Different Words maybe oranges the' for i in range(100,200)]
placeholder_data_prod = {'processed_text': prod_texts, 'file extension': np.random.choice(['.msg', '.eml'], 200), 'number of tokens': np.random.randint(500, 2000, 200), 'token bucket': ['501-1000'] * 200, 'FileModifiedTime': pd.to_datetime(pd.date_range('2023-06-01', periods=200, freq='D')), 'LOB': np.random.choice(['HR', 'Operations'], 200)} # Prod data in 2023 only
train_df = pd.DataFrame(placeholder_data)
test_df = pd.DataFrame(placeholder_data).copy(); test_df['processed_text'] = ['Test text a document with apples', 'Test b about evaluation and oranges', 'Test specific Words here too maybe bananas the']; test_df['LOB'] = ['Finance', 'HR', 'Finance']; test_df['RCC'] = ['ClassA', 'ClassB', 'ClassB']
oot_df = pd.DataFrame(placeholder_data_oot)
prod_df = pd.DataFrame(placeholder_data_prod)
val_df = train_df.sample(frac=0.5, random_state=1); val_df['processed_text'] = val_df['processed_text'] + ' validation extra word'; val_df['FileModifiedTime'] = val_df['FileModifiedTime'] - pd.Timedelta(days=180)
# --- END OF PLACEHOLDER DATA ---

all_dataframes = { 'train': train_df, 'val' : val_df, 'test': test_df, 'oot': oot_df, 'prod': prod_df }

# --- Define parameters for the function call ---
TEXT_COLUMN = 'processed_text'
TARGET_COLUMN = 'RCC'
COMMON_DISCRETE_COLS = ['file extension', 'token bucket']
COMMON_CONTINUOUS_COLS = ['number of tokens']
SPECIFIC_DISCRETE_COLS = ['LOB']
SPECIFIC_DATETIME_COLS = ['FileModifiedTime']
REFERENCE_DF_NAME = 'train'
BASE_SAVE_DIRECTORY = "./eda_outputs_final_v9" # Example save path
COMPARISON_PAIRS = [('oot', 'prod'), ('train', 'test'), ('train', 'val')]

# Call the comprehensive EDA function with new parameters
comprehensive_nlp_eda(
    dataframes=all_dataframes,
    text_col=TEXT_COLUMN,
    target_col=TARGET_COLUMN,
    common_meta_discrete=COMMON_DISCRETE_COLS,
    common_meta_continuous=COMMON_CONTINUOUS_COLS,
    specific_meta_discrete=SPECIFIC_DISCRETE_COLS,
    specific_meta_datetime=SPECIFIC_DATETIME_COLS,
    oov_reference_df_name=REFERENCE_DF_NAME,
    base_save_path=BASE_SAVE_DIRECTORY,
    high_dpi=120,
    label_rotation=45,
    max_categories_plot=40,
    plot_width=20,
    year_bucket_threshold=3,
    year_bucket_size=1,
    analyze_text_content=True,
    analyze_ngrams=True,
    top_n_terms=15,
    analyze_bigrams=True,
    ngram_analysis_sample_size=500,
    ttr_min_samples_per_class = 5,
    specific_comparisons=COMPARISON_PAIRS
)
