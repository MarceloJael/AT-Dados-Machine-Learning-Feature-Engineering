import pandas as pd
from sklearn.datasets import load_breast_cancer

# Exercicio 2.A
dados = load_breast_cancer()
df = pd.DataFrame(dados.data, columns=dados.feature_names)

print("\nExercicio 2.A:")
print(df.columns.tolist())

# Exercicio 2.B
df["mean radius (bin_fixed)"] = pd.cut(df["mean radius"], bins=4, labels=False)
df["mean area (bin_fixed)"] = pd.cut(df["mean area"], bins=4, labels=False)

print("\nExercicio 2.B:")
print(df[["mean radius", "mean radius (bin_fixed)", "mean area", "mean area (bin_fixed)"]].head(10))

# Exercicio 2.C
df["mean radius (bin_quantile)"] = pd.qcut(df["mean radius"], q=4, labels=False)
df["mean area (bin_quantile)"] = pd.qcut(df["mean area"], q=4, labels=False)

print("\nExercicio 2.C:")
print(df[["mean radius", "mean radius (bin_quantile)", "mean area", "mean area (bin_quantile)"]].head(10))
