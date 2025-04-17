import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42  # for consistent language detection results

def perform_eda(df, text_column='text', target_column=None, date_column='date',
                categorical_columns=['file_extension', 'line_of_business']):
    sns.set(style="whitegrid")

    # 1. BASIC INFO
    print("Data Info:")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Stats:\n", df.describe(include='all'))

    # 2. TEXT COLUMN ANALYSIS
    print("\nText Column Analysis:")

    df['text_length'] = df[text_column].astype(str).apply(len)
    df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))

    # Text length distribution
    plt.figure(figsize=(12, 5))
    sns.histplot(df['text_length'], bins=50, kde=True, color='purple')
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.show()

    # Word count distribution
    plt.figure(figsize=(12, 5))
    sns.histplot(df['word_count'], bins=50, kde=True, color='green')
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()

    # Most common words
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text

    corpus = df[text_column].dropna().astype(str).apply(preprocess_text).str.cat(sep=' ')
    word_freq = Counter(corpus.split())
    common_words = word_freq.most_common(30)

    # Barplot of common words
    plt.figure(figsize=(12, 6))
    words, freqs = zip(*common_words)
    sns.barplot(x=list(freqs), y=list(words), palette='magma')
    plt.title('Top 30 Most Common Words')
    plt.xlabel('Frequency')
    plt.show()

    # WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(corpus)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud of Text')
    plt.show()

    # N-gram analysis (bi-grams)
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(df[text_column].dropna().astype(str))
    bag_of_words = vec.transform(df[text_column].dropna().astype(str))
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]

    plt.figure(figsize=(12, 6))
    bigrams, freqs = zip(*words_freq)
    sns.barplot(x=list(freqs), y=list(bigrams), palette='viridis')
    plt.title('Top 20 Most Common Bigrams')
    plt.xlabel('Frequency')
    plt.show()

    # 3. TARGET VARIABLE EXPLORATION
    if target_column and target_column in df.columns:
        print(f"\nTarget Variable Distribution: {target_column}")
        plt.figure(figsize=(10, 5))
        order = df[target_column].value_counts().index
        sns.countplot(data=df, x=target_column, order=order, palette='Set2')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        plt.show()

        # Text length by class
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=target_column, y='text_length')
        plt.title('Text Length by Target Class')
        plt.xticks(rotation=45)
        plt.show()

    # 4. LANGUAGE DETECTION
    def detect_language_safe(text):
        try:
            return detect(text)
        except LangDetectException:
            return "error"

    print("\nDetecting Languages... (this may take a bit)")
    df['language'] = df[text_column].dropna().astype(str).apply(detect_language_safe)

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='language', order=df['language'].value_counts().index, palette='coolwarm')
    plt.title('Detected Language Distribution')
    plt.xlabel('Language Code')
    plt.ylabel('Count')
    plt.show()

    # 5. DATE ANALYSIS
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df['year_month'] = df[date_column].dt.to_period('M')
        plt.figure(figsize=(14, 6))
        df['year_month'].value_counts().sort_index().plot(kind='bar')
        plt.title('Document Counts Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45)
        plt.show()

    # 6. CATEGORICAL COLUMN DISTRIBUTIONS
    for col in categorical_columns:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            order = df[col].value_counts().iloc[:20].index
            sns.countplot(data=df, x=col, order=order, palette='pastel')
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.ylabel('Count')
            plt.show()

    # 7. Text length by categorical columns
    for cat in categorical_columns:
        if cat in df.columns:
            plt.figure(figsize=(12, 5))
            sns.boxplot(data=df, x=cat, y='text_length')
            plt.title(f'Text Length Distribution by {cat}')
            plt.xticks(rotation=45)
            plt.show()

    print("\nEDA complete.")

# Example usage
perform_eda(df, text_column='text', target_column='label', date_column='date',
            categorical_columns=['file_extension', 'line_of_business'])



names = ['Alice', 'Bob', 'Charlie']
pattern = r'\b(' + '|'.join(map(re.escape, names)) + r')\b'

df['text'] = df['text'].str.replace(pattern, '', regex=True)

# remove repeated consecutive words in a string
df['text'] = df['text'].str.replace(r'\b(\w+)( \1\b)+', r'\1', regex=True)

"""(\w+) captures a word

( \1)+ matches repeated occurrences of that word

r'\1' keeps just the first occurrence"""


labels = ['0-10', '11-20', '21-30', '31-50', '51-100', '100+']

df['token_bucket'] = pd.cut(df['num_tokens'], bins=bins, labels=labels, right=True)

name_set = set(name.lower() for name in names.words())

df['text'] = df['text'].apply(
    lambda x: ' '.join([word for word in str(x).split() if word.lower() not in name_set])
)


def process_in_batches(df, batch_size=10000):
    processed_batches = []
    num_batches = (len(df) + batch_size - 1) // batch_size  # ceil division

    for i, start in enumerate(tqdm(range(0, len(df), batch_size), total=num_batches)):
        end = start + batch_size
        print(f"\nProcessing batch {i + 1}/{num_batches} (rows {start} to {end - 1})")

        batch = df.iloc[start:end].copy()
        processed_batch = preprocess(batch)
        processed_batches.append(processed_batch)

    return pd.concat(processed_batches, ignore_index=True)


email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'
df['text'] = df['text'].str.replace(email_pattern, ' ', regex=True)





