# Importar as bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")


# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)

# Selecionar as colunas de interesse
df = df[['project_name', 'author_name', 'committer_name', 'is_merge']]
# Selecionar as colunas de interesse
df_selected = df[['project_name', 'author_name', 'is_merge']]
# Transformar em transações (um conjunto de colunas binárias para cada atributo)
df_encoded = pd.get_dummies(df)

# Aplicar o algoritmo Apriori
# Seleciona 10.000 linhas aleatórias
df_sample = df.sample(frac=0.01, random_state=42)  # Usa 10% dos dados
df_encoded = pd.get_dummies(df_sample)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Gerar as regras de associação
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Exibir os conjuntos frequentes e as regras de associação
print("Conjuntos Frequentes:\n", frequent_itemsets)
print("\nRegras de Associação:\n", rules)
