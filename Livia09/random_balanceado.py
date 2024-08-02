import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\Amor\OneDrive\Área de Trabalho\Projeto II\Qualidade-de-Projeto-2\Datasets\arquivo_final.csv")

# Remover colunas não necessárias (se for o caso)
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Verificar colunas com valores categóricos
print(df.dtypes)

# Substituir valores nulos pelo valor acima
df.fillna(method='ffill', inplace=True)

# Converter colunas categóricas para numéricas
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Separar dados da classe minoritária e majoritária
df_majority = df[df['Engenheirado'] == 1]
df_minority = df[df['Engenheirado'] == 0]

# Aumentar a classe minoritária
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # amostrar com substituição
                                 n_samples=len(df_majority),    # para igualar o número de registros da classe majoritária
                                 random_state=123) # para reprodutibilidade

# Combinar dados majoritários com minoritários aumentados
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Verificar a contagem das novas classes balanceadas
print(Counter(df_balanced['Engenheirado']))

# Continuar com a modelagem
X_balanced = df_balanced.drop('Engenheirado', axis=1)
y_balanced = df_balanced['Engenheirado']

# Remover a feature dominante
X_balanced_reduced = X_balanced.drop(['project_name'], axis=1)

# Dividir o dataset em conjunto de treinamento e teste
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced_reduced, y_balanced, stratify=y_balanced, random_state=2, test_size=0.3)

# Treinar e avaliar o modelo de árvore de decisão novamente
dt_bal = DecisionTreeClassifier()
dt_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = dt_bal.predict(X_test_bal)
accuracy_bal = accuracy_score(y_test_bal, y_pred_bal)
print(f'Acurácia do modelo (balanceado, feature reduzida): {accuracy_bal:.2f}')

# Validação cruzada
scores_bal = cross_val_score(dt_bal, X_balanced_reduced, y_balanced, cv=5)
print(f'Acurácia média na validação cruzada (balanceado, feature reduzida): {scores_bal.mean():.2f}')

# Treinar e avaliar o modelo Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf_clf.predict(X_test_bal)
accuracy_rf = accuracy_score(y_test_bal, y_pred_rf)
print(f'Acurácia do Random Forest (balanceado, feature reduzida): {accuracy_rf:.2f}')

# Validação cruzada com Random Forest
scores_rf = cross_val_score(rf_clf, X_balanced_reduced, y_balanced, cv=5)
print(f'Acurácia média na validação cruzada (Random Forest, balanceado, feature reduzida): {scores_rf.mean():.2f}')

# Analisar importância das features no Random Forest
importances_rf = rf_clf.feature_importances_
feature_names_rf = X_balanced_reduced.columns

feature_importances_rf = pd.DataFrame(importances_rf, index=feature_names_rf, columns=['Importance']).sort_values(by='Importance', ascending=False)
print(feature_importances_rf)

# Plotar as importâncias das features
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances_rf['Importance'], y=feature_importances_rf.index)
plt.title('Importância das Features no Random Forest')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.show()
