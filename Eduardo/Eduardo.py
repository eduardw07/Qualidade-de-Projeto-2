import pandas as pd

df = pd.read_csv(r"C:\Users\Eduardo\Desktop\Projeto Integrador 2\arquivo_final.csv")

print(df.isnull().sum())
