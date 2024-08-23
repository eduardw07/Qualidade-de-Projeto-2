import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from tabulate import tabulate

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

# Substituir valores nulos pelo valor da linha anterior
df.fillna(method='ffill', inplace=True)

# Separando os dados e o rótulo que você deseja balancear
X = df.drop(columns=['Engenheirado'])
y = df['Engenheirado']

# Aplicando o oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Reconstruindo o DataFrame balanceado
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='Engenheirado')], axis=1)

# Criar tabela de contingência entre as variáveis de interesse ('in_main', 'is_merge', 'Engenheirado')
cross_tab_absolute = pd.crosstab(df_resampled['is_merge'], [df_resampled['in_main'], df_resampled['Engenheirado']])

# Verifique o número de colunas geradas
print("Número de colunas na crosstab:", cross_tab_absolute.shape[1])
print("Colunas geradas:", cross_tab_absolute.columns)

# Ajuste o renomeamento das colunas com base nas colunas reais geradas
if cross_tab_absolute.shape[1] == 2:
    cross_tab_absolute.columns = pd.MultiIndex.from_tuples([
        ('isMain', 'Engenheirado'),
        ('isMain', 'Não Engenheirado')
    ])
elif cross_tab_absolute.shape[1] == 4:
    cross_tab_absolute.columns = pd.MultiIndex.from_tuples([
        ('isMain', 'Engenheirado'),
        ('isMain', 'Não Engenheirado'),
        ('not isMain', 'Engenheirado'),
        ('not isMain', 'Não Engenheirado')
    ])

# Exibir a tabela de contagem absoluta de forma organizada
print("Contagem Absoluta:")
print(tabulate(cross_tab_absolute, headers='keys', tablefmt='fancy_grid'))

# Normalizando para obter a contagem relativa
cross_tab_relative = cross_tab_absolute.div(cross_tab_absolute.sum().sum())

# Exibir a tabela de contagem relativa de forma organizada
print("\nContagem Relativa:")
print(tabulate(cross_tab_relative, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))


import pandas as pd
from scipy.stats import chi2_contingency

# Considerando que já temos o DataFrame df_resampled
# Criar a tabela de contingência entre 'Engenheirado' e 'is_merge'
contingency_table = pd.crosstab(df_resampled['Engenheirado'], df_resampled['is_merge'])

# Realizar o teste qui-quadrado
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Exibir os resultados
print("Tabela de Contingência:")
print(contingency_table)
print("\nValores Esperados:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
print(f"\nValor do Qui-Quadrado: {chi2:.4f}")
print(f"p-valor: {p:.4f}")
print(f"Graus de Liberdade: {dof}")

# Verificando se rejeitamos ou não a hipótese nula
alpha = 0.05
if p < alpha:
    print("\nResultado: Rejeitamos a hipótese nula (H0). Há uma associação significativa entre as variáveis 'Engenheirado' e 'is_merge'.")
else:
    print("\nResultado: Não rejeitamos a hipótese nula (H0). Não há evidências suficientes para afirmar que existe uma associação entre as variáveis 'Engenheirado' e 'is_merge'.")
