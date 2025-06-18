import pandas as pd
import os

csv_path = r"powerlifting-database\openpowerlifting.csv"
df = pd.read_csv(csv_path)


# Exercício 7.A
df_ohe = pd.get_dummies(df[['Sex', 'Equipment', 'Division']], drop_first=False)
print("\nResultado 7.A:")
print(df_ohe.head())

# Exercício 7.B
df_dummy = pd.get_dummies(df[['Sex', 'Equipment', 'Division']], drop_first=True)
print("\nResultado 7.B:")
print(df_dummy.head())

# Exercício 7.C
df_effect = df[['Sex', 'Equipment']].copy()
df_effect['Sex_Effect'] = df_effect['Sex'].map({'M': 1, 'F': -1})

equipment_types = df_effect['Equipment'].dropna().unique()
reference = equipment_types[0]

for eq in equipment_types:
    if eq != reference:
        df_effect[f'Eq_{eq}'] = df_effect['Equipment'].apply(
            lambda x: 1 if x == eq else (-1 if x == reference else 0)
        )

print("\nResultado 7.C:")
print(df_effect.head())