
import pandas as pd

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

# Substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)


print(df.value_counts())

# Seleção das variáveis relevantes
columns_of_interest = ['in_main', 'is_merge', 'Engenheirado']


# Função para calcular a contagem relativa
def relative_count(df, column):
    count = df[column].value_counts()
    relative_count = count / count.sum()
    return relative_count


# Criação da tabela de contagem relativa para cada coluna
relative_counts = {col: relative_count(df, col) for col in columns_of_interest}

# Transformando em DataFrame para melhor visualização
relative_counts_df = pd.DataFrame(relative_counts)

# Exibindo a tabela de contagem relativa
print(relative_counts_df)


# Função para calcular a contagem absoluta
def absolute_count(df, column):
    return df[column].value_counts()

# Criação da tabela de contagem absoluta para cada coluna
absolute_counts = {col: absolute_count(df, col) for col in columns_of_interest}

# Transformando em DataFrame para melhor visualização
absolute_counts_df = pd.DataFrame(absolute_counts)

# Exibindo a tabela de contagem absoluta
print(absolute_counts_df)


# Exemplo: Análise cruzada entre 'Engenheirado' e 'is_merge'
cross_tab = pd.crosstab(df['Engenheirado'], df['is_merge'])
print(cross_tab)
