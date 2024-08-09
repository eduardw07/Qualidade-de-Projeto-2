from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final1\arquivo_final.csv")

# substituir valores nulos para o valor acima
df.fillna(method='ffill', inplace=True)

X = df.drop('')
# Inicializa o vetor TF-IDF
vectorizer = TfidfVectorizer()

# Ajusta o modelo e transforma o corpus em uma matriz TF-IDF
X = vectorizer.fit_transform(df)

# Mostra os termos e seus índices
print("Features (palavras únicas):", vectorizer.get_feature_names_out())

# Exibe a matriz TF-IDF
print("Matriz TF-IDF:")
print(X.toarray())

# não teve como implementar o metodo tf-idf, pois ele para avaliar a importancia de uma palavra em um documento, e nosso projeto não tem nada haver com isso