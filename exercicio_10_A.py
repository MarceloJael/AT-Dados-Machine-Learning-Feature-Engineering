import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

csv_path = r"lung-cancer-dataset\lung-cancer-dataset.csv"
df = pd.read_csv(csv_path)

df_numeric = df.select_dtypes(include=["number"]).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o')
plt.title("Resultado")
plt.xlabel("Componentes")
plt.ylabel("Variância")
plt.grid(True)
plt.tight_layout()
plt.show()