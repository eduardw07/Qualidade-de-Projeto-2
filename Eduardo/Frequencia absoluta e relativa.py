import pandas as pd
from sklearn.utils import resample
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Leitura do arquivo CSV
df = pd.read_csv(r"../Datasets/arquivo_final.csv")

# Substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)

#print(df.value_counts())


# Supondo que seu dataset esteja no DataFrame df e que a coluna alvo seja 'is_engineered'

# Separar as classes majoritária e minoritária
majority_class = df[df['Engenheirado'] == 1]  # Classe majoritária (segue boas práticas)
minority_class = df[df['Engenheirado'] == 0]  # Classe minoritária (não segue boas práticas)

# Realizar o undersampling na classe majoritária
majority_class_downsampled = resample(majority_class,
                                      replace=False,     # Não fazer reposição
                                      n_samples=len(minority_class),  # Igualar ao tamanho da classe minoritária
                                      random_state=42)   # Reproduzibilidade

# Concatenar as classes balanceadas
df_balanced = pd.concat([majority_class_downsampled, minority_class])

# Embaralhar as amostras
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Exibir o novo balanceamento das classes
print(df_balanced['Engenheirado'].value_counts())

# Frequência absoluta
freq_absoluta = df_balanced['Engenheirado'].value_counts()

# Frequência relativa
freq_relativa = df_balanced['Engenheirado'].value_counts(normalize=True) * 100

# Exibir as tabelas
print("\nFrequência Absoluta:")
print(freq_absoluta)
print("\nFrequência Relativa (%):")
print(freq_relativa)

# Tabela de Frequência Absoluta: número de commits por autor
freq_absoluta = df_balanced.groupby('author_name')['commit_hash'].count().reset_index()
freq_absoluta.columns = ['author_name', 'commit_count']  # Renomear colunas para clareza

# Exibir a tabela de frequência absoluta
print("Tabela de Frequência Absoluta:")
print(freq_absoluta)

# Tabela de Frequência Relativa: proporção de commits por autor
total_commits = freq_absoluta['commit_count'].sum()
freq_relativa = freq_absoluta.copy()
freq_relativa['commit_percentage'] = freq_relativa['commit_count'] / total_commits

# Exibir a tabela de frequência relativa
print("\nTabela de Frequência Relativa:")
print(freq_relativa)

import pandas as pd

# Supondo que df_balanced seja o seu DataFrame já balanceado

# Frequência absoluta: Contar o número de commits por 'author_name'
freq_abs = df_balanced.groupby('author_name').size().reset_index(name='commit_count')

# Frequência relativa: Calcular a proporção de commits para cada autor em relação ao total de commits
total_commits = freq_abs['commit_count'].sum()
freq_abs['commit_percentage'] = freq_abs['commit_count'] / total_commits

# Exibir os resultados
print("Tabela de Frequência Absoluta:")
print(freq_abs[['author_name', 'commit_count']])

print("\nTabela de Frequência Relativa:")
print(freq_abs[['author_name', 'commit_percentage']])


import pandas as pd
import matplotlib.pyplot as plt

# Supondo que df_balanced já esteja carregado e limpo
# Contagem de commits por autor
author_commit_counts = df_balanced['author_name'].value_counts().reset_index()
author_commit_counts.columns = ['author_name', 'commit_count']

# Plotar gráfico de barras
plt.figure(figsize=(12, 8))
plt.bar(author_commit_counts['author_name'].head(20), author_commit_counts['commit_count'].head(20))
plt.xlabel('Autor')
plt.ylabel('Número de Commits')
plt.title('Número de Commits por Autor (Top 20)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Calcular a média de boas práticas por autor
author_practice_distribution = df_balanced.groupby('author_name')['Engenheirado'].mean().reset_index()
author_practice_distribution.columns = ['author_name', 'avg_practice']

# Plotar gráfico de barras
plt.figure(figsize=(12, 8))
plt.bar(author_practice_distribution['author_name'].head(20), author_practice_distribution['avg_practice'].head(20))
plt.xlabel('Autor')
plt.ylabel('Média de Boas Práticas')
plt.title('Distribuição de Boas Práticas por Autor (Top 20)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Supondo que df_balanced já esteja carregado
# Exemplo de estrutura do DataFrame df_balanced:
# df_balanced = pd.read_csv('caminho/para/df_balanced.csv')

# Contar o número de commits por autor e se são boas práticas ou más práticas
author_practices = df_balanced.groupby(['author_name', 'Engenheirado']).size().unstack(fill_value=0)

# Filtrar os 10 e 20 autores com mais commits
top_10_authors = author_practices.sum(axis=1).nlargest(10).index
top_20_authors = author_practices.sum(axis=1).nlargest(20).index

# Filtrar os dados para os 10 e 20 autores
top_10_data = author_practices.loc[top_10_authors]
top_20_data = author_practices.loc[top_20_authors]

# Plotar gráfico para os 10 autores
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

top_10_data.plot(kind='bar', stacked=True, ax=ax[0], color=['red', 'green'])
ax[0].set_title('Top 10 Autores com Mais Commits')
ax[0].set_xlabel('Autor')
ax[0].set_ylabel('Número de Commits')
ax[0].legend(['Más Práticas (0)', 'Boas Práticas (1)'])

# Plotar gráfico para os 20 autores
top_20_data.plot(kind='bar', stacked=True, ax=ax[1], color=['red', 'green'])
ax[1].set_title('Top 20 Autores com Mais Commits')
ax[1].set_xlabel('Autor')
ax[1].set_ylabel('Número de Commits')
ax[1].legend(['Más Práticas (0)', 'Boas Práticas (1)'])

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Exibir gráficos
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Supondo que df_balanced já esteja carregado
# Exemplo de estrutura do DataFrame df_balanced:
# df_balanced = pd.read_csv('caminho/para/df_balanced.csv')

# Contar o número de projetos em que cada autor está envolvido
author_projects = df_balanced.groupby('author_name')['project_name'].nunique()

# Filtrar os 10 autores que participaram de mais projetos distintos
top_10_authors = author_projects.nlargest(10)

# Filtrar os 20 autores que participaram de mais projetos distintos
top_20_authors = author_projects.nlargest(20)

# Criar gráfico para os 10 autores que participaram de mais projetos distintos
plt.figure(figsize=(12, 8))
top_10_authors.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Autores Envolvidos em Mais Projetos Distintos')
plt.xlabel('Número de Projetos')
plt.ylabel('Autor')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Criar gráfico para os 20 autores que participaram de mais projetos distintos
plt.figure(figsize=(12, 8))
top_20_authors.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Top 20 Autores Envolvidos em Mais Projetos Distintos')
plt.xlabel('Número de Projetos')
plt.ylabel('Autor')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Supondo que df_balanced já esteja carregado
# Exemplo de estrutura do DataFrame df_balanced:
# df_balanced = pd.read_csv('caminho/para/df_balanced.csv')

# Contar o número de projetos em que cada autor está envolvido
author_projects = df_balanced.groupby('author_name')['project_name'].nunique()

# Filtrar os 10 autores que participaram de mais projetos distintos
top_10_authors = author_projects.nlargest(10).index

# Filtrar os 20 autores que participaram de mais projetos distintos
top_20_authors = author_projects.nlargest(20).index

# Filtrar df_balanced para os autores dos top 10 e top 20
filtered_df_10 = df_balanced[df_balanced['author_name'].isin(top_10_authors)]
filtered_df_20 = df_balanced[df_balanced['author_name'].isin(top_20_authors)]

# Contar a quantidade de projetos bons e ruins por autor para os top 10
summary_10 = filtered_df_10.groupby(['author_name', 'Engenheirado']).size().unstack(fill_value=0)
summary_10 = summary_10.rename(columns={0: 'Projetos_Ruins', 1: 'Projetos_Bons'})
summary_10['Total'] = summary_10['Projetos_Bons'] + summary_10['Projetos_Ruins']

# Contar a quantidade de projetos bons e ruins por autor para os top 20
summary_20 = filtered_df_20.groupby(['author_name', 'Engenheirado']).size().unstack(fill_value=0)
summary_20 = summary_20.rename(columns={0: 'Projetos_Ruins', 1: 'Projetos_Bons'})
summary_20['Total'] = summary_20['Projetos_Bons'] + summary_20['Projetos_Ruins']


# Função para plotar gráficos com valores absolutos
def plot_author_project_quality(df_summary, title):
    df_summary = df_summary.reset_index()

    plt.figure(figsize=(14, 7))

    # Gráfico para projetos bons
    plt.subplot(1, 2, 1)
    plt.bar(df_summary['author_name'], df_summary['Projetos_Bons'], color='green')
    plt.xlabel('Autor')
    plt.ylabel('Quantidade de Projetos Bons')
    plt.title(f'{title} - Projetos Bons')
    plt.xticks(rotation=90)

    # Gráfico para projetos ruins
    plt.subplot(1, 2, 2)
    plt.bar(df_summary['author_name'], df_summary['Projetos_Ruins'], color='red')
    plt.xlabel('Autor')
    plt.ylabel('Quantidade de Projetos Ruins')
    plt.title(f'{title} - Projetos Ruins')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


# Plotar gráficos para os top 10 e top 20 autores
plot_author_project_quality(summary_10, 'Qualidade dos Projetos dos Top 10 Autores')
plot_author_project_quality(summary_20, 'Qualidade dos Projetos dos Top 20 Autores')
