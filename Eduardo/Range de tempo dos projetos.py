"""import os
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

project_root_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório atual do script
metainfo_dir = os.path.join(project_root_dir, "..", "Datasets", "metainfo")  # Caminho ajustado

if not os.path.exists(metainfo_dir):
    raise FileNotFoundError(f"O diretório {metainfo_dir} não foi encontrado.")

project_info = []

for project_dir in os.listdir(metainfo_dir):
    project_path = os.path.join(metainfo_dir, project_dir)

    if os.path.isdir(project_path):  # Confirma que é um diretório
        for file in os.listdir(project_path):
            if file.endswith(".csv"):
                csv_path = os.path.join(project_path, file)

                df = pd.read_csv(csv_path)

                if 'committer_date' in df.columns and 'author_date' in df.columns:
                    try:
                        df['committer_date'] = pd.to_datetime(df['committer_date'])
                        df['author_date'] = pd.to_datetime(df['author_date'])

                        start_date = min(df['committer_date'].min(), df['author_date'].min())
                        end_date = max(df['committer_date'].max(), df['author_date'].max())
                        duration = (end_date - start_date).days

                        if 'Engineered_ML_Project' in df.columns:
                            engineered_ml_project = df['Engineered_ML_Project'].iloc[0]
                        else:
                            engineered_ml_project = 'N/A'  # Caso a coluna não exista

                        project_info.append({
                            'project_name': df['project_name'].iloc[0],
                            'duration_days': duration,
                            'Engineered_ML_Project': engineered_ml_project
                        })
                    except Exception as e:
                        print(f"Erro ao calcular duração para {file}: {e}")
                        continue
                else:
                    print(f"As colunas 'committer_date' e/ou 'author_date' não foram encontradas em {file}.")
                    continue

df_projects = pd.DataFrame(project_info)

X = df_projects[['project_name', 'duration_days']]
y = df_projects['Engineered_ML_Project']

y = y.replace('N/A', 'N')

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=['project_name', 'duration_days'])
df_resampled['Engineered_ML_Project'] = y_resampled

print("\nDistribuição das classes no dataset balanceado:")
print(df_resampled['Engineered_ML_Project'].value_counts())

df_projects_sorted = df_resampled.sort_values(by='duration_days', ascending=False)

top_10_slowest = df_projects_sorted.head(10)
print("Top 10 projetos mais demorados:")
print(top_10_slowest)

top_10_fastest = df_projects_sorted.tail(10)
print("\nTop 10 projetos mais rápidos:")
print(top_10_fastest)

plt.figure(figsize=(12, 6))
colors_slowest = ['red' if x == 'Y' else 'blue' for x in top_10_slowest['Engineered_ML_Project']]
plt.barh(top_10_slowest['project_name'], top_10_slowest['duration_days'], color=colors_slowest)
plt.xlabel('Duração (dias)')
plt.ylabel('Nome do Projeto')
plt.title('Top 10 Projetos Mais Demorados')
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Engineered ML Project (Y)'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Não Engineered ML Project (N)')],
           loc='best')
plt.gca().invert_yaxis()  # Inverte a ordem para mostrar o maior no topo
plt.show()

plt.figure(figsize=(12, 6))
colors_fastest = ['red' if x == 'Y' else 'blue' for x in top_10_fastest['Engineered_ML_Project']]
plt.barh(top_10_fastest['project_name'], top_10_fastest['duration_days'], color=colors_fastest)
plt.xlabel('Duração (dias)')
plt.ylabel('Nome do Projeto')
plt.title('Top 10 Projetos Mais Rápidos')
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Engineered ML Project (Y)'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Não Engineered ML Project (N)')],
           loc='best')
plt.gca().invert_yaxis()  # Inverte a ordem para mostrar o menor no topo
plt.show()

plt.figure(figsize=(12, 6))
df_engineered_ml = df_resampled.groupby(['Engineered_ML_Project'])['duration_days'].mean()
df_engineered_ml.plot(kind='bar', color=['orange', 'purple'])
plt.xlabel('Engineered ML Project')
plt.ylabel('Duração Média (dias)')
plt.title('Duração Média dos Projetos por Tipo (Y/N)')
plt.show()
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

project_root_dir = os.path.dirname(os.path.abspath(__file__))
metainfo_dir = os.path.join(project_root_dir, "..", "Datasets", "metainfo")

if not os.path.exists(metainfo_dir):
    raise FileNotFoundError(f"O diretório {metainfo_dir} não foi encontrado.")

project_info = []

for project_dir in os.listdir(metainfo_dir):
    project_path = os.path.join(metainfo_dir, project_dir)

    if os.path.isdir(project_path):
        for file in os.listdir(project_path):
            if file.endswith(".csv"):
                csv_path = os.path.join(project_path, file)

                df = pd.read_csv(csv_path)

                if 'committer_date' in df.columns and 'author_date' in df.columns:
                    try:
                        df['committer_date'] = pd.to_datetime(df['committer_date'])
                        df['author_date'] = pd.to_datetime(df['author_date'])
                        start_date = min(df['committer_date'].min(), df['author_date'].min())
                        end_date = max(df['committer_date'].max(), df['author_date'].max())
                        duration = (end_date - start_date).days
                        num_commits = df.shape[0]
                        if 'Engineered_ML_Project' in df.columns:
                            engineered_ml_project = df['Engineered_ML_Project'].iloc[0]
                        else:
                            engineered_ml_project = 'N/A'
                        project_info.append({
                            'project_name': df['project_name'].iloc[0],
                            'duration_days': duration,
                            'num_commits': num_commits,
                            'Engineered_ML_Project': engineered_ml_project
                        })
                    except Exception as e:
                        print(f"Erro ao calcular duração e número de commits para {file}: {e}")
                        continue
                else:
                    print(f"As colunas 'committer_date' e/ou 'author_date' não foram encontradas em {file}.")
                    continue

df_projects = pd.DataFrame(project_info)

X = df_projects[['project_name', 'duration_days', 'num_commits']]
y = df_projects['Engineered_ML_Project']

y = y.replace('N/A', 'N')

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=['project_name', 'duration_days', 'num_commits'])
df_resampled['Engineered_ML_Project'] = y_resampled

print("\nDistribuição das classes no dataset balanceado:")
print(df_resampled['Engineered_ML_Project'].value_counts())

df_projects_sorted = df_resampled.sort_values(by='duration_days', ascending=False)

top_10_slowest = df_projects_sorted.head(10)
print("Top 10 projetos mais demorados:")
print(top_10_slowest)
top_10_fastest = df_projects_sorted.tail(10)
print("\nTop 10 projetos mais rápidos:")
print(top_10_fastest)

plt.figure(figsize=(14, 7))
colors_slowest = ['red' if x == 'Y' else 'blue' for x in top_10_slowest['Engineered_ML_Project']]
plt.barh(top_10_slowest['project_name'], top_10_slowest['duration_days'], color=colors_slowest)
plt.xlabel('Duração (dias)')
plt.ylabel('Nome do Projeto')
plt.title('Top 10 Projetos Mais Demorados')
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Engineered ML Project (Y)'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Não Engineered ML Project (N)')],
           loc='best')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(14, 7))
colors_fastest = ['red' if x == 'Y' else 'blue' for x in top_10_fastest['Engineered_ML_Project']]
plt.barh(top_10_fastest['project_name'], top_10_fastest['duration_days'], color=colors_fastest)
plt.xlabel('Duração (dias)')
plt.ylabel('Nome do Projeto')
plt.title('Top 10 Projetos Mais Rápidos')
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Engineered ML Project (Y)'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Não Engineered ML Project (N)')],
           loc='best')
plt.gca().invert_yaxis()  # Inverte a ordem para mostrar o menor no topo
plt.show()
top_10_commits = df_resampled.sort_values(by='num_commits', ascending=False).head(10)
plt.figure(figsize=(14, 7))
colors_commits = ['red' if x == 'Y' else 'blue' for x in top_10_commits['Engineered_ML_Project']]
plt.barh(top_10_commits['project_name'], top_10_commits['num_commits'], color=colors_commits)
plt.xlabel('Número de Commits')
plt.ylabel('Nome do Projeto')
plt.title('Top 10 Projetos com Mais Commits')
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Engineered ML Project (Y)'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Não Engineered ML Project (N)')],
           loc='best')
plt.gca().invert_yaxis()
plt.show()
plt.figure(figsize=(14, 7))
colors = ['red' if x == 'Y' else 'blue' for x in df_projects_sorted['Engineered_ML_Project']]
plt.scatter(df_projects_sorted['num_commits'], df_projects_sorted['duration_days'], c=colors, alpha=0.7)
plt.xlabel('Número de Commits')
plt.ylabel('Duração (dias)')
plt.title('Número de Commits vs Duração dos Projetos')
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Engineered ML Project (Y)'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Não Engineered ML Project (N)')],
           loc='best')
plt.show()
