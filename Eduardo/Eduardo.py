import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Leitura do arquivo CSV
df = pd.read_csv(r"C:\Users\Eduardo\Documents\Projeto Integrador 2\arquivo_final.csv", on_bad_lines='skip')

# Verificar e remover linhas com valores ausentes na coluna combined_text
df['combined_text'] = df['author_name'] + ' ' + df['committer_name']
df.dropna(subset=['combined_text'], inplace=True)

# Inicialize o vetor TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Transforme o texto combinado em uma matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

# Converta a matriz TF-IDF para um DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Anexe outras colunas ao DataFrame TF-IDF
result_df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# Exibir o DataFrame resultante
print(result_df)
