import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# Set high-DPI and Seaborn styling globally
plt.rcParams['figure.dpi'] = 150
sns.set(style="whitegrid", font_scale=1.1)

def run_eda(df, text_col='text', target_col='label', metadata_cols=None, top_n_tokens=20, name=''):
    metadata_cols = metadata_cols or []
    print(f"\n====== EDA Report: {name or 'Dataset'} ======\n")
    
    # 1. Dataset Overview
    print("→ Shape:", df.shape)
    print("→ Columns:", df.columns.tolist())
    print("\n→ Missing Values:\n", df.isnull().sum())

    # 2. Target Distribution
    print("\n→ Class Distribution:")
    print(df[target_col].value_counts(normalize=True).round(3))

    # 3. Text Stats
    df['n_tokens'] = df[text_col].str.split().str.len()
    df['n_chars'] = df[text_col].str.len()
    print("\n→ Text Length Stats:")
    print(df[['n_tokens', 'n_chars']].describe().round(2))

    # 4. Most Common Tokens
    vec = CountVectorizer(stop_words='english', max_features=top_n_tokens)
    bow = vec.fit_transform(df[text_col].dropna())
    token_counts = np.asarray(bow.sum(axis=0)).flatten()
    tokens_df = pd.DataFrame({'token': vec.get_feature_names_out(), 'count': token_counts})
    print(f"\n→ Top {top_n_tokens} tokens:\n", tokens_df.sort_values('count', ascending=False))

    # 5. Metadata Analysis
    for col in metadata_cols:
        if col in df.columns:
            print(f"\n→ {col} Distribution:")
            print(df[col].value_counts(normalize=True).round(3))

    # 6. Visualizations
    try:
        # Class Distribution
        plt.figure(figsize=(12, 5))
        sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index, palette="viridis")
        plt.xticks(rotation=90)
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.show()

        # Token Count Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(df['n_tokens'], bins=30, kde=False, color="skyblue")
        plt.title("Token Count Distribution")
        plt.xlabel("Number of Tokens")
        plt.tight_layout()
        plt.show()

        # Tokens per Document by Class (Boxplot)
        sample_df = df if len(df) < 100000 else df.sample(50000, random_state=42)
        plt.figure(figsize=(14, 6))
        sns.boxplot(x=target_col, y='n_tokens', data=sample_df, palette="pastel", showfliers=False)
        plt.xticks(rotation=90)
        plt.title("Tokens per Document by Class")
        plt.tight_layout()
        plt.show()

        # Avg Token Length per Class
        avg_tokens = df.groupby(target_col)['n_tokens'].mean().sort_values(ascending=False)
        plt.figure(figsize=(12, 5))
        sns.barplot(x=avg_tokens.index, y=avg_tokens.values, palette="Blues_d")
        plt.xticks(rotation=90)
        plt.title("Average Number of Tokens per Class")
        plt.ylabel("Avg Tokens")
        plt.tight_layout()
        plt.show()

        # Heatmap: Class vs Metadata (e.g., file_extension, business_division)
        for col in ['file_extension', 'business_division']:
            if col in df.columns:
                pivot = pd.crosstab(df[col], df[target_col], normalize='index')
                plt.figure(figsize=(12, 6))
                sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5)
                plt.title(f"Class Distribution by {col}")
                plt.xlabel("Class")
                plt.ylabel(col)
                plt.tight_layout()
                plt.show()

        # Time Series: Documents per Month (if applicable)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index()
            plt.figure(figsize=(10, 4))
            monthly_counts.plot(kind='line', marker='o')
            plt.title("Documents per Month")
            plt.ylabel("Document Count")
            plt.xlabel("Month")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Skipping plots due to: {e}")

    # Clean up
    df.drop(columns=['n_tokens', 'n_chars'], inplace=True, errors='ignore')
    print(f"\n====== End of Report: {name or 'Dataset'} ======\n")


run_eda(train_df,
        text_col='text',
        target_col='label',
        metadata_cols=['date', 'file_extension', 'num_tokens', 'business_division'],
        top_n_tokens=25,
        name='Training Set')
