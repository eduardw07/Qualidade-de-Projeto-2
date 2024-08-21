# Importar as bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)
pd.set_option('display.max_columns', None)



# Selecionar colunas de interesse
df_reduced = df[['project_name', 'author_name', 'committer_name', 'is_merge', 'Engenheirado']]

# Criar uma combinação de strings, mas simplificando
df_reduced['commit_combination'] = df_reduced.apply(lambda row: f"{row['project_name']}_{row['author_name']}_{row['committer_name']}_merge{row['is_merge']}_eng{row['Engenheirado']}", axis=1)

# Contagem das transações
df_reduced = df_reduced.groupby('commit_combination').size().reset_index(name='counts')

# Filtrar transações que ocorrem com menor frequência (por exemplo, pelo menos 10 vezes)
df_filtered = df_reduced[df_reduced['counts'] >= 10]

# Aplicar One-hot Encoding
df_hot_encoded = df_filtered['commit_combination'].str.get_dummies(sep='_')

# Aplicar Apriori
frequent_itemsets = apriori(df_hot_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Visualizar as regras
print(rules)
print(frequent_itemsets)



import matplotlib.pyplot as plt

# Gráfico de Dispersão (Support vs Confidence)
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.title('Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()

# Gráfico de Dispersão (Support vs Lift)
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['lift'], alpha=0.5, color='red')
plt.title('Support vs Lift')
plt.xlabel('Support')
plt.ylabel('Lift')
plt.show()


# Top 10 Regras por Lift
top_10_lift = rules.nlargest(10, 'lift')

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_10_lift)), top_10_lift['lift'], align='center', color='green')
plt.yticks(range(len(top_10_lift)), top_10_lift['antecedents'].apply(lambda x: ', '.join(list(x))))
plt.xlabel('Lift')
plt.title('Top 10 Regras por Lift')
plt.gca().invert_yaxis()
plt.show()


# Transformando Antecedents em String para Criar a Matriz
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))

# Criando uma Tabela Cruzada (Matriz de Frequências)
matrix = rules.pivot(index='antecedents_str', columns='consequents', values='lift')

# Gerando a Matriz de Calor
plt.figure(figsize=(12, 8))
sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Matriz de Calor das Regras de Associação (Lift)')
plt.show()
