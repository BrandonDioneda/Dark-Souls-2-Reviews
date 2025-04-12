import pandas as pd
import numpy as np
from top2vec import Top2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import spacy

# Load spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Game release milestones
ds2_start   = pd.Timestamp("2014-03-11")
dlc1_start  = pd.Timestamp("2014-07-22")
dlc2_start  = pd.Timestamp("2014-08-26")
dlc3_start  = pd.Timestamp("2014-09-30")
sotfs_start = pd.Timestamp("2015-04-01")

# Custom stopwords to remove common game-related terms
custom_stopwords = set([
    'dark', 'souls', 'soul', 'ds', 'scholar', 'first',
    'sin', 'edition', 'game'
])

# ----------- TEXT CLEANING -----------

def preprocess_spacy(doc):
    doc = nlp(doc.lower())
    return ' '.join([
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and token.lemma_ not in custom_stopwords
    ])

# ----------- DATA LOADING & CLEANING -----------

def load_reviews():
    vanilla = pd.read_csv('reviews.csv')
    scholar = pd.read_csv('ds2-reviews.csv')
    return pd.concat([scholar, vanilla], ignore_index=True)

def clean_reviews(df):
    df = df[df.language == 'english'].dropna(subset=['review'])
    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    df = df.set_index('recommendationid')

    df['update_date'] = pd.to_datetime(df['update_date'], unit='s')
    df['init_date']   = pd.to_datetime(df['init_date'], unit='s')

    df['review'] = (
        df['review']
        .str.lower()
        .str.replace(r'http\S+', '', regex=True)  # Remove URLs
        .str.replace(r'[^a-z]', ' ', regex=True)  # Only letters
        .str.replace(r'\b([a-z])\b', '', regex=True)  # Remove single chars
        .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        .str.strip()
    )

    df['review'] = df['review'].apply(preprocess_spacy)
    return df

def split_versions(df):
    scholar = df[df['update_date'] >= sotfs_start]
    vanilla = df[df['update_date'] < sotfs_start]
    return vanilla, scholar

def get_data():
    reviews = load_reviews()
    reviews = clean_reviews(reviews)
    vanilla, scholar = split_versions(reviews)
    return reviews, vanilla, scholar

# ----------- TOPIC MODELING -----------

def train_topic_model(docs):
    model = Top2Vec(
        documents=docs,
        speed="learn",
        workers=8
    )
    return model

def topic_clouds(topic_words, word_scores, topic_nums):
    for i in range(len(topic_words)):
        word_freq = {w: s for w, s in zip(topic_words[i], word_scores[i])}
        wc = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic {topic_nums[i]}")
        plt.show()

# ----------- COHERENCE METRICS -----------

def calculate_coherence(model, texts, topics, dictionary):
    cm = CoherenceModel(
        model=model,
        texts=texts,
        topics=topics,
        dictionary=dictionary,
        coherence='c_v'
    )
    return "Coherence Score:", cm.get_coherence()

# ----------- TIMELINE UTIL -----------

def extra_content_release(date):
    if date < ds2_start:
        return date
    if date < dlc1_start:
        return ds2_start
    elif date < dlc2_start:
        return dlc1_start
    elif date < dlc3_start:
        return dlc2_start
    elif date < sotfs_start:
        return dlc3_start
    else:
        return sotfs_start