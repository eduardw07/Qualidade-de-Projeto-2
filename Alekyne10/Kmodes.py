import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")



'''# Exibir colunas e informações básicas
print(df.columns)
print(df.isnull().sum())
print(df.shape)
print(df.info())
print(df.value_counts())
'''

### Pré-processamento dos dados

# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)



# Seleção de colunas relevantes (removendo colunas não necessárias para clustering)
df_cluster = df[['project_name', 'author_name', 'committer_name', 'in_main', 'is_merge', 'Engenheirado']]

# Convertendo todas as colunas para string para evitar problemas com k-modes
df_cluster = df_cluster.astype(str)

# Aplicação do K-Modes para clustering
km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(df_cluster)

# Adicionando os clusters ao DataFrame original
df['cluster'] = clusters

# Exibindo os resultados
print(df.head())
print(df['cluster'].value_counts())

# Função para plotar gráficos de barras para visualizar categorias dentro dos clusters
def plot_category_distribution(df, column, cluster_column='cluster'):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=column, hue=cluster_column, data=df, palette='viridis')
    plt.title(f'Distribution of {column} by Cluster')
    plt.xticks(rotation=90)
    plt.show()

# Função para plotar boxplots para variáveis numéricas dentro dos clusters
def plot_boxplot(df, column, cluster_column='cluster'):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cluster_column, y=column, data=df, palette='viridis')
    plt.title(f'Boxplot of {column} by Cluster')
    plt.show()

# Visualização das categorias
plot_category_distribution(df, 'project_name')
plot_category_distribution(df, 'author_name')
plot_category_distribution(df, 'committer_name')

# Visualização das variáveis numéricas com boxplots
plot_boxplot(df, 'in_main')
plot_boxplot(df, 'is_merge')
plot_boxplot(df, 'Engenheirado')
