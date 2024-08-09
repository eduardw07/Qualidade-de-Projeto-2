import pandas as pd

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

print(df.columns)
print(df.isnull().sum())
print(df.shape)
print(df.info())
print(df.value_counts())

