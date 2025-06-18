import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

dados = load_breast_cancer()
X = pd.DataFrame(dados.data, columns=dados.feature_names)

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

df_minmax = pd.DataFrame(X_minmax, columns=X.columns)
print("\nResultado 4.A:")
print(df_minmax.head())

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

df_std = pd.DataFrame(X_std, columns=X.columns)
print("\nResultado 4.B:")
print(df_std.head())

normalizer = Normalizer(norm='l2')
X_l2 = normalizer.fit_transform(X)

df_l2 = pd.DataFrame(X_l2, columns=X.columns)
print("\nResultado 4.C:")
print(df_l2.head())

pipeline = make_pipeline(MinMaxScaler(), Normalizer(norm='l2'))
X_combined = pipeline.fit_transform(X)