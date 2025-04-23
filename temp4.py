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
import os   # Added for path operations

# Text Analysis Imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
try:
    import umap # Preferred for embeddings
    HAS_UMAP = True
except ImportError:
    print("UMAP not found, falling back to t-SNE for embedding visualization (pip install umap-learn)")
    from sklearn.manifold import TSNE
    HAS_UMAP = False
from scipy.sparse import vstack # To stack sparse matrices

# (Rest of the imports and helper functions remain the same)
# ... basic_tokenizer, plot_discrete_distribution, compare_discrete_distributions, plot_top_ngrams ...


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
    base_save_path=None, # ADDED: Base path to save plots/outputs
    high_dpi=150,
    label_rotation=45,
    max_categories_plot=40,
    plot_width=18,
    year_bucket_threshold=15,
    year_bucket_size=2,
    # Text Analysis Params
    top_n_terms=20,
    analyze_bigrams=True,
    embedding_sample_size=5000,
    tfidf_max_features=5000
):
    """
    Performs comprehensive EDA for multiclass NLP classification on multiple dataframes.
    (Args description mostly same, added base_save_path)
    Args:
        ... (previous args) ...
        base_save_path (str, optional): Base directory path to save specific outputs (like n-grams per class).
                                         If None, these outputs are not saved.
        ... (remaining args) ...
    """
    # --- Setup ---
    plt.rcParams['figure.dpi'] = high_dpi
    print("="*80); print("Comprehensive NLP EDA Report"); print("="*80)
    target_dfs = {}

    # --- Sections 1 to 6 ---
    # (Code remains the same as the previous version)
    # ... (Basic Info, Discrete Meta, Continuous Meta, Datetime Meta, OOV, Cross-Feature) ...
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
    print("\n" + "="*80); print("--- 2. Metadata Analysis: Discrete Columns ---")
    print(f"\n--- Target Column ('{target_col}') Analysis ---")
    if target_dfs:
        for name, df in target_dfs.items(): plot_discrete_distribution(df, target_col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
        compare_discrete_distributions(target_dfs, target_col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
    else: print(f"Target column '{target_col}' not found or all NaN.")
    print("\n--- Common Discrete Metadata Analysis ---")
    for col in common_meta_discrete:
        print(f"\nAnalyzing: {col}")
        for name, df in dataframes.items():
            if col in df.columns: plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
            else: print(f"Column '{col}' not found in DataFrame '{name}'.")
        compare_discrete_distributions(dataframes, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
    print("\n--- Specific Discrete Metadata Analysis ---")
    for col in specific_meta_discrete:
        print(f"\nAnalyzing: {col}"); specific_dfs = {name: df for name, df in dataframes.items() if col in df.columns}
        if specific_dfs:
            valid_specific_dfs = {n: d for n, d in specific_dfs.items() if not d[col].isnull().all()}
            if valid_specific_dfs:
                for name, df in valid_specific_dfs.items(): plot_discrete_distribution(df, col, name, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
                compare_discrete_distributions(valid_specific_dfs, col, rotation=label_rotation, max_categories=max_categories_plot, fixed_width=plot_width)
            else: print(f"Column '{col}' found but contains only NaN values.")
        else: print(f"Column '{col}' not found.")
    print("\n" + "="*80); print("--- 3. Metadata Analysis: Continuous Columns ---")
    for col in common_meta_continuous:
        print(f"\nAnalyzing: {col}")
        num_dfs_with_col = sum(1 for df in dataframes.values() if col in df.columns and not df[col].isnull().all())
        if num_dfs_with_col > 0:
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
        plot_data_boxplot = []
        for name, df in dataframes.items():
             if col in df.columns and not df[col].isnull().all():
                 temp_df = df[[col]].dropna().copy();
                 if not temp_df.empty: temp_df['DataFrame'] = name; plot_data_boxplot.append(temp_df)
        if plot_data_boxplot:
             fig, ax = plt.subplots(figsize=(10, 6)); combined_df_boxplot = pd.concat(plot_data_boxplot, ignore_index=True)
             sns.boxplot(x='DataFrame', y=col, data=combined_df_boxplot, palette='viridis', showfliers=False, ax=ax)
             ax.set_title(f'Comparison of "{col}" Distribution (Outliers Hidden)'); ax.set_xlabel('DataFrame'); ax.set_ylabel(col); plt.show()
        print(f"\n--- Distribution of '{col}' by Target ('{target_col}') ---")
        target_dfs_with_col = {name: df for name, df in target_dfs.items() if col in df.columns and not df[col].isnull().all()}
        if target_dfs_with_col:
            for name, df in target_dfs_with_col.items():
                if not df[target_col].isnull().all():
                    df_plot = df[[col, target_col]].dropna().copy()
                    if df_plot.empty: print(f"Skipping box plot for {name}."); continue
                    df_plot[target_col] = df_plot[target_col].astype(str); target_order = sorted(df_plot[target_col].unique())
                    fig, ax = plt.subplots(figsize=(plot_width, 7));
                    sns.boxplot(x=target_col, y=col, data=df_plot, palette='viridis', order=target_order, showfliers=False, ax=ax)
                    ax.set_title(f'{name}: Dist of "{col}" by "{target_col}" (Outliers Hidden)'); ax.set_xlabel(target_col); ax.set_ylabel(col); ax.tick_params(axis='x', rotation=rotation)
                    is_positive = (df_plot[col] > 0) if pd.api.types.is_numeric_dtype(df_plot[col]) else pd.Series(False, index=df_plot.index)
                    if is_positive.all() and col == 'number of tokens' and (df_plot[col].max() / df_plot[col].median() > 50):
                        ax.set_yscale('log'); ax.set_ylabel(f"{col} (Log Scale)"); print(f"Applied log scale for {name}.")
                    elif col == 'number of tokens' and not is_positive.all() and is_positive.any(): print(f"Note: Log scale not applied for {name} (non-positive values).")
                    fig.tight_layout(); plt.show()
                else: print(f"Target column in '{name}' NaN.")
        else: print(f"Could not perform analysis by target for '{col}'.")
    print("\n" + "="*80); print("--- 4. Metadata Analysis: Datetime Columns ---")
    for col in specific_meta_datetime: # Logic unchanged
         print(f"\nAnalyzing: {col}"); dt_dfs = {}
         for name, df in dataframes.items():
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
         if dt_dfs: # Bucketing and plotting unchanged
             for name, df_dt in dt_dfs.items():
                  print(f"\n--- Datetime Analysis for {col} in {name} ---"); df_dt_nonan = df_dt.dropna(subset=[col]).copy()
                  if df_dt_nonan.empty: print(f"No valid datetime values."); continue
                  df_dt_nonan['Year'] = df_dt_nonan[col].dt.year; yearly_counts = df_dt_nonan['Year'].value_counts().sort_index(); unique_years = yearly_counts.index.astype(int)
                  if not yearly_counts.empty:
                      num_years = len(unique_years); plot_title = f'{name}: Document Count'; x_label = 'Year'
                      if num_years > year_bucket_threshold:
                          print(f"Bucketing years ({num_years} > {year_bucket_threshold})."); min_year, max_year = unique_years.min(), unique_years.max(); actual_bucket_size = max(1, year_bucket_size)
                          bins = list(range(min_year, max_year + actual_bucket_size, actual_bucket_size));
                          if len(bins)>1 and bins[-1] <= max_year : bins.append(max_year + 1)
                          elif len(bins)==1: bins.append(max_year + 1)
                          labels = [f"{bins[i]}-{bins[i+1]-1}" if bins[i+1]-1 > bins[i] else f"{bins[i]}" for i in range(len(bins)-1)]
                          df_dt_nonan['Year Bucket'] = pd.cut(df_dt_nonan['Year'], bins=bins, labels=labels, right=False, include_lowest=True)
                          counts_to_plot = df_dt_nonan['Year Bucket'].value_counts().sort_index(); plot_title += f' per {actual_bucket_size}-Year Bucket'; x_label = f'{actual_bucket_size}-Year Bucket'
                      else: counts_to_plot = yearly_counts; plot_title += ' per Year'
                      fig, ax = plt.subplots(figsize=(plot_width, 6)); counts_to_plot.plot(kind='bar', color=sns.color_palette('viridis', len(counts_to_plot)), ax=ax)
                      ax.set_title(plot_title + f' ({col})'); ax.set_xlabel(x_label); ax.set_ylabel('Number of Documents'); ax.tick_params(axis='x', rotation=rotation); ax.grid(True, axis='y'); fig.tight_layout(); plt.show()
                  else: print(f"No non-NaN data points for yearly counts.")
         else: print(f"Column '{col}' not found or unusable.")
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
            if valid_oov: fig, ax = plt.subplots(figsize=(max(6, len(valid_oov)*1.5), 5)); s = pd.Series(valid_oov).sort_values(); sns.barplot(x=s.index, y=s.values, palette='viridis', ax=ax); ax.set_title(f'OOV % (Token vs. "{oov_reference_df_name}")'); ax.set_ylabel('OOV %'); ax.tick_params(axis='x', rotation=rotation, ha='right'); fig.tight_layout(); plt.show()
            if valid_unique_oov: fig, ax = plt.subplots(figsize=(max(6, len(valid_unique_oov)*1.5), 5)); s = pd.Series(valid_unique_oov).sort_values(); sns.barplot(x=s.index, y=s.values, palette='magma', ax=ax); ax.set_title(f'OOV % (Unique Word vs. "{oov_reference_df_name}")'); ax.set_ylabel('OOV %'); ax.tick_params(axis='x', rotation=rotation, ha='right'); fig.tight_layout(); plt.show()
    print("\n" + "="*80); print("--- 6. Cross-Feature Analysis (Examples) ---")
    col1_ex1 = common_meta_discrete[0] if common_meta_discrete else None; col2_ex1 = common_meta_continuous[0] if common_meta_continuous else None
    if col1_ex1 and col2_ex1: # Example 1 uses showfliers=False
        print(f"\n--- Analyzing: '{col1_ex1}' vs '{col2_ex1}' ---"); data = []
        for name, df in dataframes.items():
            if col1_ex1 in df.columns and col2_ex1 in df.columns:
                 if not df[col1_ex1].isnull().all() and not df[col2_ex1].isnull().all():
                     tdf = df[[col1_ex1, col2_ex1]].dropna().copy()
                     if not tdf.empty: tdf['DataFrame'] = name; tdf[col1_ex1] = tdf[col1_ex1].astype(str); data.append(tdf)
        if data:
            cdf = pd.concat(data, ignore_index=True); order = cdf[col1_ex1].value_counts().index[:max_categories_plot]
            fig, ax = plt.subplots(figsize=(plot_width, 7)); sns.boxplot(x=col1_ex1, y=col2_ex1, hue='DataFrame', data=cdf, palette='viridis', order=order, showfliers=False, ax=ax)
            ax.set_title(f'Relationship: "{col1_ex1}" vs "{col2_ex1}" (Outliers Hidden)'); ax.set_xlabel(col1_ex1); ax.set_ylabel(col2_ex1); ax.tick_params(axis='x', rotation=rotation); ax.legend(title='DataFrame', bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
        else: print(f"Not enough data.")
    else: print("\nSkipping Cross-Feature Ex1.")
    col1_ex2 = common_meta_discrete[0] if common_meta_discrete else None
    if col1_ex2 and target_col: # Example 2 unchanged
        print(f"\n--- Analyzing: '{col1_ex2}' vs '{target_col}' ---")
        dfs = {name: df for name, df in target_dfs.items() if col1_ex2 in df.columns and not df[col1_ex2].isnull().all() and not df[target_col].isnull().all()}
        if dfs:
            for name, df in dfs.items():
                 dfp = df[[col1_ex2, target_col]].dropna().copy();
                 if dfp.empty: continue
                 dfp[col1_ex2] = dfp[col1_ex2].astype(str); dfp[target_col] = dfp[target_col].astype(str)
                 try:
                    ct = pd.crosstab(dfp[col1_ex2], dfp[target_col], normalize='index') * 100; order = dfp[col1_ex2].value_counts().index[:max_categories_plot]; ct = ct.reindex(order).dropna(how='all')
                    if ct.empty: continue
                    fig, ax = plt.subplots(figsize=(plot_width, 7)); ct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax); ax.set_title(f'{name}: Proportion of "{target_col}" within "{col1_ex2}" (Top {len(order)} Cats)'); ax.set_xlabel(col1_ex2); ax.set_ylabel('%'); ax.tick_params(axis='x', rotation=rotation); ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()
                 except Exception as e: print(f"Error plotting stacked bar for {name}: {e}")
        else: print(f"Not enough data.")
    elif not col1_ex2: print("\nSkipping Cross-Feature Ex2.")


    # --- 7. Text Content Analysis ---
    print("\n" + "="*80); print("--- 7. Text Content Analysis ---")

    # 7a. Type-Token Ratio (TTR) - Code unchanged
    print(f"\n--- 7a. Type-Token Ratio (Lexical Diversity) ---"); ttr_results = {}
    print("\nCalculating TTR per Dataset:")
    for name, df in dataframes.items():
        if text_col not in df.columns or df[text_col].isnull().all(): print(f"- {name}: Skipping"); ttr_results[f"{name}_overall"] = np.nan; continue
        token_lists = df[text_col].dropna().apply(basic_tokenizer); all_tokens = [token for sublist in token_lists for token in sublist]
        if not all_tokens: print(f"- {name}: Skipping (No tokens)"); ttr_results[f"{name}_overall"] = np.nan; continue
        total_tokens = len(all_tokens); unique_tokens = len(set(all_tokens)); ttr = (unique_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        ttr_results[f"{name}_overall"] = ttr; print(f"- {name}: Unique={unique_tokens}, Total={total_tokens}, TTR={ttr:.2f}%")
    print(f"\nCalculating TTR per Class ('{oov_reference_df_name}' data):")
    ref_df_ttr = dataframes.get(oov_reference_df_name)
    if ref_df_ttr is not None and target_col in ref_df_ttr.columns and text_col in ref_df_ttr.columns:
        grouped = ref_df_ttr.dropna(subset=[text_col, target_col]).groupby(target_col)
        for class_label, group_df in grouped:
            token_lists = group_df[text_col].dropna().apply(basic_tokenizer); all_tokens = [token for sublist in token_lists for token in sublist]
            if not all_tokens: print(f"- Class '{class_label}': Skipping (No tokens)"); ttr_results[f"Class_{class_label}"] = np.nan; continue
            total_tokens = len(all_tokens); unique_tokens = len(set(all_tokens)); ttr = (unique_tokens / total_tokens) * 100 if total_tokens > 0 else 0
            ttr_results[f"Class_{class_label}"] = ttr; print(f"- Class '{class_label}': Unique={unique_tokens}, Total={total_tokens}, TTR={ttr:.2f}%")
    else: print(f"Could not calculate TTR per class.")
    valid_ttr = {k: v for k,v in ttr_results.items() if pd.notna(v)}
    if valid_ttr:
        ttr_series = pd.Series(valid_ttr).sort_values(); fig, ax = plt.subplots(figsize=(max(8, len(ttr_series)*0.6), 5)); sns.barplot(x=ttr_series.index, y=ttr_series.values, palette='coolwarm', ax=ax)
        ax.set_ylabel("TTR (%)"); ax.set_title("Type-Token Ratio Comparison"); ax.tick_params(axis='x', rotation=60, ha='right'); fig.tight_layout(); plt.show()

    # --- 7b. Top N-grams per Class (MODIFIED: Added tqdm) ---
    print(f"\n--- 7b. Top N-grams per Class (using '{oov_reference_df_name}' data) ---")
    ref_df_ngram = dataframes.get(oov_reference_df_name)
    if ref_df_ngram is not None and target_col in ref_df_ngram.columns and text_col in ref_df_ngram.columns:
        grouped = ref_df_ngram.dropna(subset=[text_col, target_col]).groupby(target_col)

        print(f"Analyzing top {top_n_terms} Unigrams (saving plots if path provided)...")
        # Added tqdm wrapper
        for class_label, group_df in tqdm(grouped, desc="Processing Unigrams per Class"):
            corpus = group_df[text_col].dropna().astype(str)
            save_filepath = None
            if base_save_path:
                safe_class_label = re.sub(r'[^\w\-]+', '_', str(class_label))
                save_dir = os.path.join(base_save_path, "ngram_analysis", "ngrams_per_class", oov_reference_df_name, safe_class_label)
                save_filename = f"top_{top_n_terms}_unigrams.png"; save_filepath = os.path.join(save_dir, save_filename)
            if not corpus.empty: plot_top_ngrams(corpus, title=f"Top {top_n_terms} Unigrams for Class: {class_label}", ngram_range=(1,1), top_n=top_n_terms, save_path=save_filepath)
            # else: print(f"Skipping Unigrams for Class '{class_label}'.") # Optionally silence this inside loop

        if analyze_bigrams:
            print(f"\nAnalyzing top {top_n_terms} Bigrams (saving plots if path provided)...")
             # Added tqdm wrapper
            for class_label, group_df in tqdm(grouped, desc="Processing Bigrams per Class"):
                 corpus = group_df[text_col].dropna().astype(str)
                 save_filepath = None
                 if base_save_path:
                     safe_class_label = re.sub(r'[^\w\-]+', '_', str(class_label))
                     save_dir = os.path.join(base_save_path, "ngram_analysis", "ngrams_per_class", oov_reference_df_name, safe_class_label)
                     save_filename = f"top_{top_n_terms}_bigrams.png"; save_filepath = os.path.join(save_dir, save_filename)
                 if not corpus.empty: plot_top_ngrams(corpus, title=f"Top {top_n_terms} Bigrams for Class: {class_label}", ngram_range=(2,2), top_n=top_n_terms, save_path=save_filepath)
                 # else: print(f"Skipping Bigrams for Class '{class_label}'.") # Optionally silence
    else: print(f"Could not calculate N-grams per class.")

    # --- 7c. Top N-grams per Dataset (MODIFIED: Show plots, no saving) ---
    # (No tqdm needed here as it already shows plots one by one)
    print(f"\n--- 7c. Top N-grams per Dataset ---")
    ngram_sample_size = min(50000, max(embedding_sample_size * 5, 10000))
    print(f"Analyzing top {top_n_terms} Unigrams per Dataset (sampling large DFs to max {ngram_sample_size})...")
    for name, df in dataframes.items(): # Logic unchanged
         if text_col not in df.columns or df[text_col].isnull().all(): print(f"Skipping Unigrams for '{name}'."); continue
         df_sampled = df;
         if len(df) > ngram_sample_size: print(f"Sampling '{name}' (size {len(df)}) to {ngram_sample_size}."); df_sampled = df.sample(n=ngram_sample_size, random_state=42)
         corpus = df_sampled[text_col].dropna().astype(str)
         if not corpus.empty: plot_top_ngrams(corpus, title=f"Top {top_n_terms} Unigrams for Dataset: {name}", ngram_range=(1,1), top_n=top_n_terms, save_path=None)
         else: print(f"Skipping Unigrams for '{name}'.")
    if analyze_bigrams:
        print(f"\nAnalyzing top {top_n_terms} Bigrams per Dataset (sampling large DFs to max {ngram_sample_size})...")
        for name, df in dataframes.items(): # Logic unchanged
             if text_col not in df.columns or df[text_col].isnull().all(): print(f"Skipping Bigrams for '{name}'."); continue
             df_sampled = df;
             if len(df) > ngram_sample_size: df_sampled = df.sample(n=ngram_sample_size, random_state=42)
             corpus = df_sampled[text_col].dropna().astype(str)
             if not corpus.empty: plot_top_ngrams(corpus, title=f"Top {top_n_terms} Bigrams for Dataset: {name}", ngram_range=(2,2), top_n=top_n_terms, save_path=None)
             else: print(f"Skipping Bigrams for '{name}'.")

    # --- 7d. Embedding Visualization (Dataset Comparison) ---
    # (Code remains the same)
    print(f"\n--- 7d. Embedding Visualization for Dataset Comparison ---")
    print(f"Using {'UMAP' if HAS_UMAP else 't-SNE'} + TF-IDF (max_feat={tfidf_max_features}). Sampling max {embedding_sample_size} docs/dataset.")
    ref_df_embed = dataframes.get(oov_reference_df_name)
    if ref_df_embed is not None and text_col in ref_df_embed.columns and not ref_df_embed[text_col].isnull().all():
        try: # Embedding Pipeline unchanged
            corpus_list = []; dataset_labels = []
            print("Preparing data for embedding...");
            for name, df in tqdm(dataframes.items(), desc="Sampling Datasets"):
                if text_col not in df.columns or df[text_col].isnull().all(): print(f"Skipping '{name}'."); continue
                df_valid = df.dropna(subset=[text_col]);
                if df_valid.empty: print(f"Skipping '{name}': No text."); continue
                df_sampled = df_valid.sample(n=min(len(df_valid), embedding_sample_size), random_state=42)
                corpus_list.extend(df_sampled[text_col].astype(str).tolist()); dataset_labels.extend([name] * len(df_sampled))
            if not corpus_list: print("No data for embedding.")
            else:
                print(f"Combined corpus size: {len(corpus_list)}")
                print(f"Fitting TF-IDF on '{oov_reference_df_name}'...")
                tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features, lowercase=False, tokenizer=lambda text: text.split())
                ref_corpus = ref_df_embed.dropna(subset=[text_col])[text_col].astype(str)
                if not ref_corpus.empty: tfidf_vectorizer.fit(ref_corpus)
                else: raise ValueError(f"Ref DF '{oov_reference_df_name}' has no text to fit TF-IDF.")
                print("Transforming combined corpus..."); tfidf_matrix = tfidf_vectorizer.transform(corpus_list)
                print("Applying dimensionality reduction...")
                if HAS_UMAP: reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, metric='cosine', low_memory=True)
                else: # t-SNE fallback unchanged
                    n_comp = 50 if tfidf_matrix.shape[1] > 50 else tfidf_matrix.shape[1]
                    if tfidf_matrix.shape[1] > 50: print("Applying TruncatedSVD..."); from sklearn.decomposition import TruncatedSVD; svd = TruncatedSVD(n_components=n_comp, random_state=42); tfidf_reduced = svd.fit_transform(tfidf_matrix)
                    else: tfidf_reduced = tfidf_matrix.toarray()
                    reducer = TSNE(n_components=2, random_state=42, perplexity=30, metric='cosine', init='random', learning_rate='auto')
                embedding = reducer.fit_transform(tfidf_matrix if HAS_UMAP else tfidf_reduced)
                print("Generating plot...")
                embedding_df = pd.DataFrame(embedding, columns=['x', 'y']); embedding_df['Dataset'] = dataset_labels
                fig, ax = plt.subplots(figsize=(12, 10)); sns.scatterplot(data=embedding_df, x='x', y='y', hue='Dataset', palette='viridis', alpha=0.6, s=20, ax=ax)
                ax.set_title(f"2D Embedding ({'UMAP' if HAS_UMAP else 't-SNE'}) Colored by Dataset"); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left'); fig.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()
        except Exception as e: print(f"Error during embedding viz: {e}"); import traceback; traceback.print_exc()
    else: print(f"Skipping embedding visualization: Ref DF '{oov_reference_df_name}' or text col missing/invalid.")


    print("\n" + "="*80); print("EDA Complete."); print("="*80)


# --- Example Function Call ---
# (Example call structure remains the same, assumes data loading happens before)

# Create placeholder DataFrames for the example call to run without error
# --- REPLACE THIS WITH YOUR ACTUAL DATAFRAMES ---
placeholder_data = {'processed_text': ['Text A Train Document', 'TEXT B ABOUT MODELS', 'Train specific Words here'], 'file extension': ['.pdf', '.docx', '.pdf'], 'number of tokens': [100, 200, 150], 'token bucket': ['100-500', '100-500', '100-500'], 'RCC': ['ClassA', 'ClassB', 'ClassA']}
placeholder_data_oot = {'processed_text': ['Text c oot version', 'OOT unique content'], 'file extension': ['.txt', '.txt'], 'number of tokens': [50, 60], 'token bucket': ['0-100', '0-100'], 'RCC': ['ClassA', 'ClassC'], 'FileModifiedTime': [pd.Timestamp('2023-01-15'), pd.Timestamp('2023-02-20')], 'LOB': ['Finance', 'Legal']}
prod_texts = [f'Prod text {i} example Content for Production' for i in range(100)] + [f'Prod text {i} Different Words' for i in range(100,200)]
placeholder_data_prod = {'processed_text': prod_texts, 'file extension': np.random.choice(['.msg', '.eml'], 200), 'number of tokens': np.random.randint(500, 2000, 200), 'token bucket': ['501-1000'] * 200, 'FileModifiedTime': pd.date_range('2022-01-01', periods=200, freq='D'), 'LOB': np.random.choice(['HR', 'Operations'], 200)}
train_df = pd.DataFrame(placeholder_data)
test_df = pd.DataFrame(placeholder_data).copy(); test_df['processed_text'] = ['Test text a document', 'Test b about evaluation', 'Test specific Words here too']
oot_df = pd.DataFrame(placeholder_data_oot)
prod_df = pd.DataFrame(placeholder_data_prod)
oot_df['FileModifiedTime'] = pd.to_datetime(oot_df['FileModifiedTime'], errors='coerce')
prod_df['FileModifiedTime'] = pd.to_datetime(prod_df['FileModifiedTime'], errors='coerce')
# --- END OF PLACEHOLDER DATA ---

all_dataframes = { 'train': train_df, 'test': test_df, 'oot': oot_df, 'prod': prod_df }

# --- Define parameters for the function call ---
TEXT_COLUMN = 'processed_text'
TARGET_COLUMN = 'RCC'
COMMON_DISCRETE_COLS = ['file extension', 'token bucket']
COMMON_CONTINUOUS_COLS = ['number of tokens']
SPECIFIC_DISCRETE_COLS = ['LOB']
SPECIFIC_DATETIME_COLS = ['FileModifiedTime']
REFERENCE_DF_NAME = 'train'
# --- SET YOUR SAVE PATH HERE ---
BASE_SAVE_DIRECTORY = "./eda_outputs_saved" # Example: Save to a subfolder

# Call the comprehensive EDA function with text analysis params
comprehensive_nlp_eda(
    dataframes=all_dataframes,
    text_col=TEXT_COLUMN,
    target_col=TARGET_COLUMN,
    common_meta_discrete=COMMON_DISCRETE_COLS,
    common_meta_continuous=COMMON_CONTINUOUS_COLS,
    specific_meta_discrete=SPECIFIC_DISCRETE_COLS,
    specific_meta_datetime=SPECIFIC_DATETIME_COLS,
    oov_reference_df_name=REFERENCE_DF_NAME,
    base_save_path=BASE_SAVE_DIRECTORY, # Pass the save path
    high_dpi=120,
    label_rotation=45,
    max_categories_plot=40,
    plot_width=20,
    year_bucket_threshold=15,
    year_bucket_size=2,
    top_n_terms=20,
    analyze_bigrams=True,
    embedding_sample_size=5000,
    tfidf_max_features=5000
)
