import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

base_path = "aclImdb/train"
categories = ["pos", "neg"]

def load_reviews():
    texts, labels = [], []
    for category in categories:
        folder = os.path.join(base_path, category)
        label = 1 if category == "pos" else 0
        for fname in os.listdir(folder)[:100]:
            with open(os.path.join(folder, fname), encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

texts, labels = load_reviews()

# Exercicio 6.A
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("\nResultado 6.A:")
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(df_tfidf.iloc[:5, :5])

# Exercicio 6.B
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
print("\nResultado 6.B:")
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))