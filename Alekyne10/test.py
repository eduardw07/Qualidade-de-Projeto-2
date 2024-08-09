# Importar as bibliotecas necessárias
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd


# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")


# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)

# Selecionar as colunas de interesse
df = df[['in_main', 'Engenheirado', 'is_merge']]
# Selecionar as colunas de interesse

df_encoded = pd.get_dummies(df)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Gerar as regras de associação
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Exibir os conjuntos frequentes e as regras de associação
print("Conjuntos Frequentes:\n", frequent_itemsets)
print("\nRegras de Associação:\n", rules)
