import pandas as pd
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


# Carregar o dataset
df_bruto = pd.read_csv('../Datasets/arquivo_final.csv')

# Remover colunas desnecessárias
df = df_bruto.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

# Verificar as primeiras linhas do dataset
print(df.head(), "\n")

# Resumo estatístico
print("\n Descrição do Dataset \n", df.describe(), "\n")

# Distribuição de classes (assumindo que a coluna 'classe' define se é bem ou mal relacionado)
print(df['Engenheirado'].value_counts())

# Verificando a % das classes
# Total de instâncias
total = df['Engenheirado'].value_counts().sum()

# Percentual de cada classe
percentual = df['Engenheirado'].value_counts(normalize=True) * 100
print("\nA porcentagem da distribuição entre as classes é de: \n", percentual)


# Verificando a existencia de valores inexistentes
print("\nDataframe sem tratamento de NA \n",df.isnull().sum())

# Retirando os NAs
df_limpo = df.dropna()
print("\nDataframe sem NA \n", df_limpo.isnull().sum())

# Contagem de commits que são merges
contagem_merges_por_classe = df_limpo[df_limpo['is_merge'] == 1].groupby('Engenheirado').size()

# Visualizar o resultado
plt.figure(figsize=(8, 6))
sns.barplot(x=contagem_merges_por_classe.index, y=contagem_merges_por_classe.values, palette='viridis')
plt.title('Contagem de Commits de Merge por Classe de Engenheirado')
plt.xlabel('Classe de Engenheirado')
plt.ylabel('Contagem de Commits de Merge')
plt.xticks(ticks=[0, 1], labels=['Más Práticas (0)', 'Boas Práticas (1)'])
plt.show()

# Análise de merges por projeto
merge_by_project = df_limpo.groupby('project_name')['is_merge'].mean()
print(" ")
print("Analise das margens do projeto:")
print(merge_by_project.sort_values(ascending=False))


# Aplicar TF-IDF em uma coluna de texto (assumindo que 'descricao' é a coluna de texto)
tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(df_limpo['project_name'])

# Transformar o resultado em um DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Calcular a média dos valores TF-IDF para cada palavra
tfidf_means = tfidf_df.mean().sort_values(ascending=False)

# Visualizar as palavras mais importantes
"""plt.figure(figsize=(12, 8))
sns.barplot(x=tfidf_means.head(20).values, y=tfidf_means.head(20).index, palette='viridis')
plt.title('Top 20 Palavras com Maior TF-IDF Médio')
plt.xlabel('TF-IDF Médio')
plt.ylabel('Palavra')
#plt.show()"""

#### BALANCEANDO O DATASET AQUI

# USANDO O METODO UNDER-SAMPLING

# Separar características e alvo
X = df_limpo.drop(columns=['project_name', 'Engenheirado'])
y = df_limpo['Engenheirado']

# Definir colunas categóricas
categorical_features = ['is_merge', 'in_main']  # Liste aqui todas as colunas categóricas

# Criar um pré-processador para One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Aplicar o pré-processamento
X_processed = preprocessor.fit_transform(X)

# Criar o modelo de under-sampling
under_sampler = RandomUnderSampler(random_state=42)

# Aplicar o under-sampling
X_resampled, y_resampled = under_sampler.fit_resample(X_processed, y)

# Adicionar a coluna 'project_name' ao dataset balanceado
df_resampled = pd.DataFrame(X_resampled, columns=preprocessor.get_feature_names_out())
df_resampled['Engenheirado'] = y_resampled.reset_index(drop=True)

# Recriar o DataFrame com a coluna 'project_name'
df_resampled = df_resampled.join(df_limpo[['project_name']].reset_index(drop=True), how='left')

# Verificar a distribuição das classes no DataFrame balanceado
print("Balanceando o Dataset com o metodo de under-samping ")
print(df_resampled['Engenheirado'].value_counts(normalize=True) * 100)

# APLICANDO O SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

## FINALMENTE ESSA ... DEU CERTO, TAVA PRA ENDOIDAR JÁ

# APLICANDO O TF-IDF

tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(df_resampled['project_name'])

# Transformar o resultado em um DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Adicionar as variáveis TF-IDF ao DataFrame balanceado
df_tfidf = pd.concat([df_resampled, tfidf_df], axis=1)

# Verificar as primeiras linhas do DataFrame final
print(df_tfidf.head())

print(df_tfidf.describe())

## PREPARANDO A MODELAGEM DOS DADOS
# Separar características e alvo
X = df_tfidf.drop(columns=['project_name', 'Engenheirado'])
y = df_tfidf['Engenheirado']

# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## AVALIANDO OS MODELOS
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

## PLOT CURVA ROC
y_probs = model.predict_proba(X_test)[:, 1]

# Calcular as curvas ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (Área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

