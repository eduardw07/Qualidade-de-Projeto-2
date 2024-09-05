import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura do arquivo CSV
df = pd.read_csv(r"../Datasets/arquivo_final.csv")

# Substituir valores nulos pelo valor acima
df.ffill(inplace=True)

# Selecionar as colunas de interesse
df = df[['project_name', 'author_name', 'committer_name', 'is_merge']]

# Seleciona uma amostra de 1% dos dados
df_sample = df.sample(frac=0.01, random_state=42)

# Aplicar o get_dummies apenas nas colunas que são realmente necessárias
df_encoded = pd.get_dummies(df_sample[['project_name', 'author_name', 'is_merge']])

# Aplicar o algoritmo Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Gerar as regras de associação
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Exibir os conjuntos frequentes e as regras de associação
print("Conjuntos Frequentes:\n", frequent_itemsets)
print("\nRegras de Associação:\n", rules)
