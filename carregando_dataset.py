import pandas as pd

#Caso tenha adicionado o csv ao seu projeto, utilize:
df = pd.read_csv("arquivo_final.csv")

#Caso nao tenha adicionado o csv ao seu projeto tire o comentario do codigo abaixo, edite e utilize-o:
#df = pd.read_csv(r"caminho_do_arquivo_csv\arquivo_final.csv")

#Retire o comentario para visualizar as colunas presentes no csv:
#print(df.columns)

#Tire o comentario para visualizar as 5 primeiras informações do csv (cabivel edição para maior visualização dos dados):
#print(df.head(5))

#Tire o comentario para visualizar uma analise descritiva dos dados:
#print(df.describe())
