import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")


# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)

df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Converter colunas categóricas para numéricas
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

X = df.drop('Engenheirado', axis= 1)
y= df['Engenheirado']

X_train, X_test,y_train, y_test = train_test_split(X, y, stratify=y, random_state=2, test_size=0.3)


# Criar um modelo de árvore de decisão
dt= DecisionTreeClassifier(max_depth=2, random_state=42, min_samples_split= 4, min_samples_leaf= 5)
dt.fit(X_train, y_train)

# Fazer previsões
y_pred = dt.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print("Árvore de Decisão:")

print(f'Acurácia do modelo: {accuracy:.2f}')
# Realizar a validação cruzada com 5 folds
scores = cross_val_score(dt, X, y, cv=5)
print(f'Acurácia média na validação cruzada: {scores.mean():.2f}')
print(classification_report(y_test, y_pred))




importances = dt.feature_importances_
feature_names = X.columns

feature_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values(by='Importance', ascending=False)
print(feature_importances)


rf_clf = RandomForestClassifier(random_state=42, n_estimators= 100, max_depth= 3)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Acurácia do Random Forest: {accuracy_rf:.2f}')

# Realizar a validação cruzada com 5 folds
scores = cross_val_score(rf_clf, X, y, cv=5)
print(f'Acurácia média na validação cruzada: {scores.mean():.2f}')


print("Random Forest:")
print(classification_report(y_test, y_pred_rf))



print(Counter(y))

# Utilização da tecnica smote para balanciamento das classes
#Métricas Adicionais: outras métricas como ROC-AUC

#Análise de Erros: Analise os casos em que os modelos erraram para a classe minoritária para entender melhor onde o
# modelo pode estar falhando.