import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from top2vec import Top2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

ds2_start = pd.Timestamp("2014-03-11")
dlc1_start = pd.Timestamp("2014-07-22")
dlc2_start = pd.Timestamp("2014-08-26")
dlc3_start = pd.Timestamp("2014-09-30")
sotfs_start = pd.Timestamp("2015-04-01")

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

def remove_stopwords_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['dark', 'souls', 'soul', 'ds', 'scholar', 'first', 'sin', 'edition', 'game'])

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
        if word not in stop_words
    ]

    return ' '.join(lemmatized_words)

def get_data():
    vanilla = pd.read_csv('reviews.csv')
    scholar = pd.read_csv('ds2-reviews.csv')
    reviews = pd.concat([scholar, vanilla], ignore_index=True)

    reviews = reviews[reviews.language == 'english'].dropna(subset=['review'])
    reviews.drop(columns={'Unnamed: 0'}, inplace=True)
    reviews = reviews.set_index('recommendationid')
    reviews.update_date   = pd.to_datetime(reviews["update_date"], unit='s')
    reviews.init_date     = pd.to_datetime(reviews["init_date"], unit='s')
    reviews['review'] = (
        reviews['review']
        .str.lower()
        .str.replace(r'http\S+', '', regex=True)  # Remove URLs
        .str.replace(r'[^a-z]', ' ', regex=True)  # Keep only letters
        .str.replace(r'\b([a-z])\b', ' ', regex=True)  # Remove single letters
        .str.replace(r'\s+', ' ', regex=True)     # Replace multiple spaces with single space
        .str.strip()
    )
    reviews['review'] = reviews.review.apply(remove_stopwords_and_lemmatize)

    scholar = reviews[reviews.update_date() >= pd.Timestamp("2015-04-01")] 
    vanilla = reviews[reviews.update_date() < pd.Timestamp("2015-04-01")] 

    return reviews, vanilla, scholar

def topics(documents):
    mdl = Top2Vec(
        documents=documents,
        # contextual_top2vec=True,
        speed="learn",
        workers=8,
    )
    return mdl

def topic_clouds(topic_words, word_scores, topic_nums):
    for i in range(len(topic_words)):
        words = topic_words[i]
        scores = word_scores[i]
        word_freq = {w: s for w, s in zip(words, scores)}
        
        wc = WordCloud(width=800, height=400, background_color='black')
        wc.generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic {topic_nums[i]}")
        plt.show()

def extra_content_release(date):
    if date < ds2_start:
        return date

    if (date >= ds2_start) and (date < dlc1_start):
        return ds2_start
    elif date < dlc2_start:
        return dlc1_start
    elif date < dlc3_start:
        return dlc2_start
    elif date < sotfs_start:
        return dlc3_start
    else:
        return sotfs_start
    
def metrics(model, text, topics, dictionary):
    coherence_model = CoherenceModel(model=model, texts=text, topics=topics, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return "Coherence Score:", coherence_score