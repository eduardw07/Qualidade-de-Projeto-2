from pydriller import Repository

# Defina o caminho para o repositório Git
repo_path = r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.git"

# Inicialize o GitRepository com o caminho do repositório
repo = Repository(repo_path)

# Lista para armazenar informações dos commits
commits_data = []

# Itera sobre todos os commits no repositório
for commit in repo.traverse_commits():
    # Extrai informações do commit
    commit_info = {
        'hash': commit.hash,
        'author': commit.author.name,
        'date': commit.committer_date,
        'message': commit.msg,
        'modified_files': [f.filename for f in commit.modifications],
        'insertions': sum(f.added for f in commit.modifications),
        'deletions': sum(f.deleted for f in commit.modifications),
        'lines': sum(f.added + f.deleted for f in commit.modifications),
        'buggy': 'fix' in commit.msg.lower()  # Exemplo: considera um commit como buggy se "fix" estiver na mensagem
    }
    commits_data.append(commit_info)

# Convertendo para DataFrame
import pandas as pd
df_commits = pd.DataFrame(commits_data)

# Exibindo o DataFrame com as informações dos commits
print(df_commits.head())

# Exemplo de análise: contar o número de commits por autor
author_commit_counts = df_commits['author'].value_counts()
print("Número de commits por autor:")
print(author_commit_counts)

# Filtrar commits que contenham a palavra "bug" na mensagem
buggy_commits = df_commits[df_commits['message'].str.contains('bug', case=False, na=False)]
print("Commits que mencionam 'bug':")
print(buggy_commits)
