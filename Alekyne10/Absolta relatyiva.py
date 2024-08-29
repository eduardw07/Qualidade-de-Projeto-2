import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

# Substituir valores nulos pelo valor da linha anterior
df.fillna(method='ffill', inplace=True)

# Separando os dados e o rótulo que você deseja balancear
X = df.drop(columns=['Engenheirado'])
y = df['Engenheirado']


# Exibir a contagem original das classes


# Aplicando o undersampling para balancear as classes
undersampler = RandomUnderSampler(sampling_strategy={0: 75661, 1: 75661}, random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Exibir a contagem após o balanceamento
print("\nContagem das classes após o balanceamento:")
print(pd.Series(y_resampled).value_counts())

# Contagem da quantidade de merges
quantidade_merges = df['is_merge'].sum()

print(f"Quantidade de merges: {quantidade_merges}")
print(df['is_merge'].value_counts())



# Reconstruindo o DataFrame balanceado
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Criar a matriz de contingência 2x2 no conjunto balanceado
matrix_2x2 = pd.crosstab(df_resampled['is_merge'], df_resampled['Engenheirado'])

# Exibir a matriz
print("Matriz de Contingência 2x2:")
print(matrix_2x2)

# Criar a matriz de contingência 3x2 no conjunto balanceado com 'is_merge', 'in_main' e 'Engenheirado'
matrix_3x2 = pd.crosstab(index=[df_resampled['is_merge'], df_resampled['in_main']], columns=df_resampled['Engenheirado'])

# Exibir a matriz
print("Matriz de Contingência 3x2:")
print(matrix_3x2)
