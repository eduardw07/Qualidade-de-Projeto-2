import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)



### Grafico de barra e boxplot
# Contar o número de commits por projeto
commit_counts = df['project_name'].value_counts()

# Criar um dataframe com o número de commits por projeto
commit_counts_df = pd.DataFrame(commit_counts).reset_index()
commit_counts_df.columns = ['project_name', 'num_commits']

# Configurar o estilo dos gráficos
sns.set(style="whitegrid")

# Criar uma visualização combinada vertical
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Boxplot
sns.boxplot(x='num_commits', data=commit_counts_df, ax=axs[0])
axs[0].set_title('Boxplot do Número de Commits por Projeto')

# Histograma
sns.histplot(commit_counts_df['num_commits'], bins=30, ax=axs[1])
axs[1].set_title('Histograma do Número de Commits por Projeto')
axs[1].set_xlabel('Número de Commits')
axs[1].set_ylabel('Frequência')

plt.tight_layout()
plt.show()


# Definir o limite de commits para o espectro
limite_commits = 1000  # Você pode ajustar este valor conforme necessário

# Classificar os projetos em dentro e fora do espectro
commit_counts_df['dentro_do_espectro'] = commit_counts_df['num_commits'] <= limite_commits

# Contar o número de projetos dentro e fora do espectro
contagem_classificacoes = commit_counts_df['dentro_do_espectro'].value_counts()
labels = ['Dentro do Espectro', 'Fora do Espectro']
sizes = contagem_classificacoes.values
colors = ['#add4d3', '#fc8d62']

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title('Diferença entre Projetos Dentro e Fora do Espectro de Commits')
plt.show()

