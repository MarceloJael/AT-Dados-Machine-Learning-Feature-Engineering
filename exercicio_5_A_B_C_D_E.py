import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

base_path = "aclImdb/train"
categories = ["pos", "neg"]

def load_reviews():
    texts, labels = [], []
    for category in categories:
        folder = os.path.join(base_path, category)
        label = 1 if category == "pos" else 0
        for fname in os.listdir(folder):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

texts, labels = load_reviews()

texts = texts[:100]
labels = labels[:100]

#Exercicio 5.A
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    tokens = text.lower().split()
    return [t for t in tokens if t.isalpha() and t not in stop_words]

cleaned_tokens = [remove_stopwords(text) for text in texts]

#Exercicio  5.B
stemmer = PorterStemmer()
stemmed = [[stemmer.stem(token) for token in tokens] for tokens in cleaned_tokens]

# Exercicio 5.C
lemmatizer = WordNetLemmatizer()
lemmatized = [[lemmatizer.lemmatize(token) for token in tokens] for tokens in cleaned_tokens]

stemmed_texts = [" ".join(tokens) for tokens in stemmed]
lemmatized_texts = [" ".join(tokens) for tokens in lemmatized]

# Exercicio 5.D
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(stemmed_texts)
df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
print("\nResultado 5.D:")
print(df_bow.head())

#Exercicio 5.E
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
X_bigram = vectorizer_bigram.fit_transform(lemmatized_texts)
df_bigram = pd.DataFrame(X_bigram.toarray(), columns=vectorizer_bigram.get_feature_names_out())
print("\nResultado 5.E:")
print(df_bigram.head())