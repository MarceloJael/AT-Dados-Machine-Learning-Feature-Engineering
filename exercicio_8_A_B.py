import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
import numpy as np

csv_path = r"powerlifting-database\openpowerlifting.csv"
df = pd.read_csv(csv_path)

data = df[['Equipment']].fillna('Unknown').astype(str).to_dict(orient='records')

hasher = FeatureHasher(input_type='dict', n_features=8)
hashed_features = hasher.fit_transform(data)

hashed_df = pd.DataFrame(hashed_features.toarray())

print("\nResultado 8.A:")
print(hashed_df.head())

le = LabelEncoder()
labels = le.fit_transform(df['Equipment'].fillna('Unknown'))

bin_counts = np.bincount(labels)

print("\nResultado 8.A:")
for i, count in enumerate(bin_counts):
    print(f"{le.inverse_transform([i])[0]}: {count}")