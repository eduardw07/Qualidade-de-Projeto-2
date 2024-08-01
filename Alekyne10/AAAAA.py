import pandas as pd
import zipfile

# Caminho para o arquivo zip
zip_file_path = r"C:\Users\MASTER\OneDrive\Área de Trabalho\Alekyne\projeto integrador\Qualidade-de-Projeto-2\Datasets\arquivo_final.part1.zip"
# Nome do arquivo CSV dentro do arquivo zip
csv_file_name = "arquivo_final.csv"

# Abre o arquivo zip
with zipfile.ZipFile(zip_file_path, 'r') as z:
    # Abre o arquivo CSV dentro do arquivo zip
    with z.open(csv_file_name) as f:
        # Lê o CSV em um DataFrame pandas
        df = pd.read_csv(f)

# Exibir as primeiras linhas do DataFrame
print(df.head())
