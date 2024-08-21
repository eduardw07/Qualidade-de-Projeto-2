from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

# Substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)

# Suponha que 'Engenheirado' seja a variável target para classificação binária
# E que 'in_main', 'is_merge' sejam features (ajuste conforme o seu dataset)

# Separando as features (X) e o target (y)
X = df[['in_main', 'is_merge']]
y = df['Engenheirado']

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo SMOTE e Random Under Sampler
smote = SMOTE(random_state=42)
undersample = RandomUnderSampler(random_state=42)

# Criando um pipeline para aplicar SMOTE seguido de Random Under Sampling
resampling_pipeline = Pipeline(steps=[
    ('smote', smote),
    ('undersample', undersample)
])

# Aplicando o pipeline para balancear os dados
X_train_res, y_train_res = resampling_pipeline.fit_resample(X_train, y_train)

# Inicializando e treinando o classificador de árvore de decisão
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train_res, y_train_res)

# Fazendo previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliação do modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))