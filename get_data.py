import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['dark', 'souls', 'soul', 'ds', 'scholar', 'first', 'sin', 'edition', 'game'])
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])

def get_data():
    df = pd.read_csv('reviews.csv')
    reviews = df.copy()
    reviews = reviews[reviews.language == 'english'].dropna(subset=['review'])
    reviews = reviews.set_index('recommendationid')
    reviews.drop(columns={'Unnamed: 0'}, inplace=True)

    reviews['month_name'] = pd.to_datetime(reviews.update_date, unit='s').dt.month_name()
    reviews['month']      = pd.to_datetime(reviews.update_date, unit='s').dt.month
    reviews['year']       = pd.to_datetime(reviews.update_date, unit='s').dt.year
    reviews['day']        = pd.to_datetime(reviews.update_date, unit='s').dt.day

    reviews['review'] = (
        reviews['review']
        .str.lower()
        .str.replace(r'http\S+', '', regex=True)  # Remove URLs
        .str.replace(r'[^a-z]', ' ', regex=True)  # Keep only letters
        .str.replace(r'\s+', ' ', regex=True)     # Replace multiple spaces with single space
        .str.strip()
    )

    reviews['review'] = reviews.review.apply(remove_stopwords)

    return reviews