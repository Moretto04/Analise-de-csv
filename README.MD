# Estrutura Logica do Codigo  

## 1. Importacao de Bibliotecas
> import pandas as pd
> import numpy as np
> import seaborn as sns
> import matplotlib.pyplot as plt
> from sklearn.preprocessing import MinMaxScaler

**Decisao Tecnica:** 

    - pandas e numpy para manipulacao de dados;
    - seaborn e matplotlib para visualizacao estatistica;
    - MinMaxScaler para normalizacao de variaveis numericas, essencial em analises quantitativas com escalas distintas.

## 2. Leitura da Base de Dados
> df = pd.read_csv('economy_and_population_of_cities_in_brazil.csv')

**Justificativa:** 

    - O uso de pd.read_csv() permite importar um arquivo em formato amplamente utilizado em analises reais, como os encontrados no IBGE ou Kaggle.

**Validacao:** 

    - Verifica os dados para ver se foram carregados corretamente com .head(), .info() e .columns.

## 3. Exploracao Inicial dos Dados
> print(df.head())
> print(df.columns.tolist())
> print(df.info())

**Objetivo:** 

    - Compreender as estruturas dos dados antes de transforma-los.

**Decisao Tecnica:** 

    - A inspecao visual e importante para detectar:
        - Colunas com nomes inadequados
        - Tipos incorretos de dados (ex: números como strings)
        - Presenca de valores ausentes

## 4. Pre-processamento e Limpeza dos Dados
### 4.1 Conversao de Coluna PIB
> df['Pib_2014'] = df['Pib_2014'].str.replace(',', '.')
> df['Pib_2014'] = pd.to_numeric(df['Pib_2014'], errors='coerce')

**Racional:**

    - Dados numericos como o PIB podem vir em formato de string com virgulas. Para processa-los corretamente, eh necessario converte-los para float.
    - Uso de errors='coerce': Garante que valores invalidos sejam tratados como NaN, que serao limpos na etapa seguinte.

### 4.2 Preenchimento de Valores Ausentes
> for coluna in colunas_numericas:
    mediana = df[coluna].median()
    df[coluna] = df[coluna].fillna(mediana)

**Decisao Tecnica:** 

    - Opta-se por preencher valores ausentes com a mediana, pois, ela eh mais robusta contra outliers do que a media.

**Evita Vies:** 

    - Manter consistencia estatistica dos dados mesmo apos limpeza.

### 4.3 Normalizacao das Variaveis
> scaler = MinMaxScaler()
> df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

**Justificativa:** 

    - As colunas como Pib_2014 e PopEstimada_2018 tem escalas muito diferentes. A normalizacao (transformar os dados para faixa 0 a 1) permite comparacoes diretas e melhora visualizacoes e analises quantitativas como correlacao.

## 5. Analise de Correlacao
> correlacao = df[colunas_numericas].corr()

**Objetivo:** 

    - Avaliar o grau de associacao entre variaveis numericas.

**Tipo de Correlacao:** 

    - A funcao corr() calcula a correlacao de Pearson, adequada para dados quantitativos continuos.

### 5.1 Visualizacao via Heatmap
> sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")

**Decisao Tecnica:**

    - annot=True: mostra valores numericos na matriz.

    - coolwarm: facilita a leitura da intensidade da correlacao.

    - Permite detectar padroes de colinearidade ou independencia entre variaveis.

# 6. Grafico de Dispersao: PIB vs Populacao
> sns.scatterplot(data=df, x='PopEstimada_2018', y='Pib_2014')

**Justificativa:**

    - Analisar visualmente a relacao entre populacao e PIB das cidades brasileiras.

**Importancia:**

    - Identifica agrupamentos, anomalias e tendencias.
    - Fundamental para levantar hipoteses como: "Cidades mais populosas geram mais PIB?"

# 7. Resumo das Decisoes Tecnicas

Decisao	                                    Justificativa
Substituicao de virgulas por pontos	        Converter texto em números reais
Uso de median() para valores ausentes	    Robusto contra outliers
Normalizacao com MinMaxScaler	            Facilita comparacoes e analises multivariadas
Analise de correlacao com Pearson	        Avaliar relacoes lineares entre variaveis continuas
Visualizacao com seaborn/matplotlib	        Clareza estatistica e estetica