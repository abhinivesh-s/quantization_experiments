import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from scipy.stats import entropy

plt.rcParams['figure.dpi'] = 150
sns.set(style="whitegrid", font_scale=1.1)

def compute_diversity_metrics(text_series):
    """Compute token diversity metrics: TTR, Hapax Ratio, Shannon Entropy"""
    tokens = ' '.join(text_series.dropna()).split()
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    hapax = sum(1 for t, c in token_counts.items() if c == 1)
    hapax_ratio = hapax / total_tokens if total_tokens else 0
    ttr = unique_tokens / total_tokens if total_tokens else 0
    probs = np.array(list(token_counts.values())) / total_tokens
    shannon_entropy = entropy(probs, base=2) if total_tokens else 0

    return {
        'Total Tokens': total_tokens,
        'Unique Tokens': unique_tokens,
        'TTR': round(ttr, 4),
        'Hapax Ratio': round(hapax_ratio, 4),
        'Shannon Entropy': round(shannon_entropy, 4)
    }

def run_eda(dfs, names, text_col='text', target_col='label', metadata_cols=None, top_n_tokens=20, heavy_viz=True, ref_vocab=None):
    metadata_cols = metadata_cols or []
    
    if len(dfs) != len(names):
        raise ValueError("The number of datasets and names must be the same.")
    
    for name, df in zip(names, dfs):
        print(f"\n====== EDA Report: {name} ======\n")
        
        # Sample for speed if needed
        if len(df) > 100_000 and heavy_viz:
            df = df.sample(50_000, random_state=42)
            print("⚠️ Using sample of 50K rows for heavy visualization")
        
        # 1. Dataset Overview
        print(f"→ {name} - Shape:", df.shape)
        print("→ Columns:", df.columns.tolist())
        print("\n→ Missing Values:\n", df.isnull().sum())

        # 2. Target Distribution (Raw + Normalized)
        print(f"\n→ {name} - Class Distribution (Raw + Normalized):")
        class_counts = df[target_col].value_counts()
        class_norm = df[target_col].value_counts(normalize=True).round(3)
        class_df = pd.DataFrame({'count': class_counts, 'proportion': class_norm})
        print(class_df)

        # 3. Text Stats
        df['n_tokens'] = df[text_col].str.split().str.len()
        df['n_chars'] = df[text_col].str.len()
        print(f"\n→ {name} - Text Length Stats:")
        print(df[['n_tokens', 'n_chars']].describe().round(2))

        # 4. Most Common Tokens
        vec = CountVectorizer(stop_words='english', max_features=top_n_tokens)
        bow = vec.fit_transform(df[text_col].dropna())
        token_counts = np.asarray(bow.sum(axis=0)).flatten()
        try:
            feature_names = vec.get_feature_names_out()
        except AttributeError:
            feature_names = vec.get_feature_names()

        tokens_df = pd.DataFrame({'token': feature_names, 'count': token_counts})
        print(f"\n→ {name} - Top {top_n_tokens} tokens:\n", tokens_df.sort_values('count', ascending=False))

        # 5. Metadata Analysis (Raw + Normalized)
        for col in metadata_cols:
            if col in df.columns:
                print(f"\n→ {name} - {col} Distribution (Raw + Normalized):")
                meta_counts = df[col].value_counts()
                meta_norm = df[col].value_counts(normalize=True).round(3)
                meta_df = pd.DataFrame({'count': meta_counts, 'proportion': meta_norm})
                print(meta_df)

        # 6. Visualizations

        try:
            # Class Distribution (Side by side for multiple datasets)
            plt.figure(figsize=(14, 6))
            for i, (name, df) in enumerate(zip(names, dfs)):
                sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index, palette="viridis", label=name)
            plt.title("Class Distribution Comparison")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Datasets")
            plt.tight_layout()
            plt.show()

            # Token Count Histogram (Side by side for multiple datasets)
            plt.figure(figsize=(14, 6))
            for i, (name, df) in enumerate(zip(names, dfs)):
                sns.histplot(df['n_tokens'], bins=30, kde=False, label=name, element='step', stat='density')
            plt.title("Token Count Distribution Comparison")
            plt.xlabel("Number of Tokens")
            plt.legend(title="Datasets")
            plt.tight_layout()
            plt.show()

            # Average Token Length per Class (Side by side for multiple datasets)
            avg_tokens_per_class = {}
            for name, df in zip(names, dfs):
                avg_tokens_per_class[name] = df.groupby(target_col)['n_tokens'].mean().sort_values(ascending=False)

            plt.figure(figsize=(14, 6))
            for name, avg_tokens in avg_tokens_per_class.items():
                sns.barplot(x=avg_tokens.index, y=avg_tokens.values, label=name)
            plt.title("Average Number of Tokens per Class Comparison")
            plt.ylabel("Avg Tokens")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Datasets")
            plt.tight_layout()
            plt.show()

            # Bigram Frequency Plot (Side by side for multiple datasets)
            for name, df in zip(names, dfs):
                bigram_vec = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=top_n_tokens)
                bigram_bow = bigram_vec.fit_transform(df[text_col].dropna())
                bigram_counts = np.asarray(bigram_bow.sum(axis=0)).flatten()
                bigram_names = bigram_vec.get_feature_names_out()
                bigram_df = pd.DataFrame({'bigram': bigram_names, 'count': bigram_counts}).sort_values('count', ascending=False)

                plt.figure(figsize=(12, 5))
                sns.barplot(data=bigram_df.head(top_n_tokens), x='bigram', y='count', label=name)
                plt.title(f"Top Bigrams - {name}")
                plt.xticks(rotation=45, ha='right')
                plt.legend(title="Datasets")
                plt.tight_layout()
                plt.show()

            # TF-IDF Embedding Clustering (Side by side for multiple datasets)
            for name, df in zip(names, dfs):
                tfidf_vec = TfidfVectorizer(stop_words='english', max_features=5000)
                tfidf_matrix = tfidf_vec.fit_transform(df[text_col].fillna(''))

                svd = TruncatedSVD(n_components=2, random_state=42)
                reduced = svd.fit_transform(tfidf_matrix)

                cluster_df = pd.DataFrame(reduced, columns=['x', 'y'])
                cluster_df['dataset'] = name

                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=cluster_df, x='x', y='y', hue='dataset', palette='tab20', legend='full', s=10)
                plt.title(f"TF-IDF SVD Clusters - {name}")
                plt.xlabel("SVD-1")
                plt.ylabel("SVD-2")
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Skipping plots due to: {e}")

        # Token Diversity Metrics (Side by side for multiple datasets)
        print(f"\n→ {name} - Token Diversity Metrics:")
        diversity = compute_diversity_metrics(df[text_col])
        for k, v in diversity.items():
            print(f"   {k}: {v}")

        # OOV Rate if ref_vocab is provided
        if ref_vocab is not None:
            oov_tokens = [t for t in ' '.join(df[text_col].dropna()).split() if t not in ref_vocab]
            oov_ratio = len(oov_tokens) / diversity['Total Tokens'] if diversity['Total Tokens'] else 0
            print(f"\n→ {name} - OOV Rate vs Reference Vocabulary: {round(oov_ratio * 100, 2)}%")

        # Clean up
        df.drop(columns=['n_tokens', 'n_chars'], inplace=True, errors='ignore')
        print(f"\n====== End of Report: {name} ======\n")


# First extract vocab from train set
train_tokens = ' '.join(train_df['text'].dropna()).split()
train_vocab = set(train_tokens)

# Run on train, test, and prod data with comparison
run_eda([train_df, test_df, prod_df], ['Train', 'Test', 'Prod'], text_col='text', target_col='label',
        metadata_cols=['file_extension', 'business_division'], name='Comparison', ref_vocab=train_vocab)
