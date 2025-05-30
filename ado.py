# 1. Importacao das bibliotecas essenciais
import pandas as pd                
import numpy as np                 
import seaborn as sns              # Visualizacao estatistica (graficos)
import matplotlib.pyplot as plt    # Visualizacao grafica
from sklearn.preprocessing import MinMaxScaler  # Ferramenta para normalizacao de dados

# Configuraçoes visuais padrão dos graficos
sns.set(style='whitegrid')             
plt.rcParams['figure.figsize'] = (12, 8)  

# 2. Leitura do arquivo CSV
df = pd.read_csv('Cities_Brazil_IBGE.csv')

# 3. Analise exploratoria inicial
print("Primeiras linhas do DataFrame:")
print(df.head())  # Mostra as primeiras linhas do dataset para inspecao visual

print("\nColunas disponiveis:")
print(df.columns.tolist())  # Lista todos os nomes das colunas disponiveis

print("\nInformacoes gerais:")
print(df.info())  # Exibe tipo de dados, valores nulos, etc.

# 4. PRE-PROCESSAMENTO DOS DADOS

# 4.1 Conversao de valores numericos formatados como texto
# Algumas colunas numericas, como 'Pib_2014', podem estar com virgulas como separador decimal
if df['Pib_2014'].dtype == 'object':
    df['Pib_2014'] = df['Pib_2014'].str.replace(',', '.')            # Substitui virgulas por ponto
    df['Pib_2014'] = pd.to_numeric(df['Pib_2014'], errors='coerce')  # Converte para float, invalidando o que for incorreto

# 4.2 Preenchimento de valores ausentes com a mediana
# A mediana porque eh robusta contra valores extremos
colunas_numericas = df.select_dtypes(include=[np.number]).columns  # Seleciona apenas colunas numericas
for coluna in colunas_numericas:
    mediana = df[coluna].median()
    df[coluna] = df[coluna].fillna(mediana)  # Substitui valores ausentes por mediana

# 4.3 Normalizacao dos dados
# Traz todos os valores para uma escala comum [0, 1] — util para analises comparativas
scaler = MinMaxScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# 5. ANALISE DE CORRELACAO ENTRE VARIAVEIS

# Calcula a matriz de correlacao de Pearson entre as variaveis numericas
correlacao = df[colunas_numericas].corr()

print("\nMatriz de correlacao:")
print(correlacao)

# 5.1 Visualizacao grafica da matriz de correlacao
# O heatmap mostra visualmente como as variaveis se relacionam entre si
plt.figure(figsize=(12, 10))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")  # annot=True exibe os valores
plt.title('Matriz de Correlacao entre Variaveis Numericas')
plt.show()

# 6. ANALISE VISUAL: RELAÇAO ENTRE PIB E POPULAÇAO

# Cria um grafico de dispersao (scatter plot) entre Populacao Estimada e PIB
if 'Pib_2014' in df.columns and 'PopEstimada_2018' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PopEstimada_2018', y='Pib_2014')  # Eixo X = populacao, Eixo Y = PIB
    plt.title('Relacao entre Populacao Estimada (2018) e PIB (2014)')
    plt.xlabel('Populacao Estimada 2018 (normalizada)')
    plt.ylabel('PIB 2014 (normalizado)')
    plt.show()
