import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\Amor\OneDrive\Área de Trabalho\Projeto II\Qualidade-de-Projeto-2\Datasets\arquivo_final.csv")


# Remover colunas não necessárias (se for o caso)
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Verificar colunas com valores categóricos
print(df.dtypes)

# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)


print(df.value_counts())

# Converter colunas categóricas para numéricas
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

X = df.drop('Engenheirado', axis= 1)
y= df['Engenheirado']

X_train, X_test,y_train, y_test = train_test_split(X, y, stratify=y, random_state=2, test_size=0.3)


# Criar um modelo de árvore de decisão
dt= DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Fazer previsões
y_pred = dt.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
from sklearn.model_selection import cross_val_score

# Realizar a validação cruzada com 5 folds
scores = cross_val_score(dt, X, y, cv=5)
print(f'Acurácia média na validação cruzada: {scores.mean():.2f}')

importances = dt.feature_importances_
feature_names = X.columns

feature_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values(by='Importance', ascending=False)
print(feature_importances)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Acurácia do Random Forest: {accuracy_rf:.2f}')

from collections import Counter
print(Counter(y))

