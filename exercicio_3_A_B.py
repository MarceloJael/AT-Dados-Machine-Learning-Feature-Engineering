import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer

dados = load_breast_cancer()
X = pd.DataFrame(dados.data, columns=dados.feature_names)

def minmax_normalize(X):
    return (X - X.min()) / (X.max() - X.min())

transformador = FunctionTransformer(minmax_normalize)
X_minmax = transformador.fit_transform(X)

df_minmax = pd.DataFrame(X_minmax, columns=X.columns)

print("Resultado 3.A:")
print(df_minmax.head())

pt = PowerTransformer()
X_power = pt.fit_transform(X)

df_power = pd.DataFrame(X_power, columns=X.columns)

print("\nResultado 3.B:")
print(df_power.head())