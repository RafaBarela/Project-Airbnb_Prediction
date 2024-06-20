# %% [markdown]
# ### Projeto Airbnb Rio de Janeiro - Ferramenta de Previsão de Preço de Imóvel para pessoas.
# ### Airbnb Rio de Janeiro Project - Property Price Forecast Tool for people.
# 

# %% [markdown]
# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Meu objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# %% [markdown]
# ### Context
# 
# On Airbnb, anyone who has a room or property of any type (apartment, house, chalet, inn, etc.) can offer their property to be rented on a daily basis.
# 
# You create your host profile (a person who offers a property for daily rental) and create your property advertisement.
# 
# In this ad, the host must describe the characteristics of the property as completely as possible, in order to help renters/travelers choose the best property for them (and in order to make their ad more attractive)
# 
# There are dozens of possible customizations in your listing, from minimum daily rate, price, number of rooms, to cancellation rules, extra fee for extra guests, requirement for landlord identity verification, etc.
# 
# ### My objective
# 
# Build a price prediction model that allows an ordinary person who owns a property to know how much they should charge for the daily rate of their property.
# 
# Or even, for the common landlord, given the property he is looking for, it helps to know whether that property has an attractive price (below the average for properties with the same characteristics) or not.
# 
# ### What we have available, inspirations and credits
# 
# The databases were taken from the kaggle website: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# You will notice similarities between the solution we will develop here and his, but also some significant differences in the project construction process.
# 
# - The databases are the prices of properties obtained and their respective characteristics in each month.
# - Prices are given in reais (R$)
# - We have databases from April 2018 to May 2020, with the exception of June 2018, which does not have a database
# 
# ### Initial Expectations
# 
# - I believe that seasonality can be an important factor, as months like December tend to be very expensive in RJ
# - The location of the property should make a big difference in the price, since in Rio de Janeiro the location can completely change the characteristics of the place (security, natural beauty, tourist attractions)
# - Additions/Amenities can have a significant impact, as we have many old buildings and houses in Rio de Janeiro
# 
# Let's find out how much these factors impact and whether we have other not so intuitive factors that are extremely important.

# %% [markdown]
# ##### Importar Bibliotecas e Base De Dados.
# ##### Import Libraries and Database.
# 

# %%
import pandas as pd
import pathlib
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# %%
meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path('dataset')

bases = []

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    bases.append(df)

base_airbnb = pd.concat(bases)    
display(base_airbnb)

# %% [markdown]
# - Como temos muitas colunas, nosso modelo pode acabar ficando muito lento.
# - Além disso, uma análise rápida permite ver que várias colunas não são necessárias para o nosso modelo de previsão, por isso, vamos excluir algumas colunas da nossa base.
# - Tipos de Colunas que vamos excluir:
#  1. Ids, Links e informações irrelevantes para o modelo.
#  2. Colunas repetidas ou muito parecidas com outra (Que dão a mesma informação para o modelo)
#  3. Colunas preenchidas com texto livres, Ex: Descrição do Ambiente
#  4. Colunas em que todos ou quase todos os valores são iguais.
# 
#  - Para isso, vamos criar um arquivo excel com os 1.000 primeiros registros e fazer uma análise qualitativa.
#       

# %% [markdown]
# - As we have many columns, our model may end up being very slow.
# - Additionally, a quick analysis lets you see that several columns are not necessary for our prediction model, so let's exclude some columns from our database.
# - Types of Columns that we will exclude:
#  1. Ids, Links and information irrelevant to the model.
#  2. Columns that are repeated or very similar to each other (which give the same information to the model)
#  3. Columns filled with free text, Ex: Environment Description
#  4. Columns in which all or almost all values ​​are the same.
# 
#  - To do this, we will create an excel file with the first 1,000 records and perform a qualitative analysis.

# %%
print(list(base_airbnb.columns))
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')

# %% [markdown]
# #### Depois da Análise qualitativa das colunas, levando em conta os critérios explicado a cima, ficamos com as seguintes colunas:
# 
# #### After the qualitative analysis of the columns, taking into account the criteria explained above, we are left with the following columns:
# 

# %%
colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
display(base_airbnb)

# %% [markdown]
# #### Tratando valores NaN
# 
# - Visualizando os dados, percebemos que existe uma grande dispariedade em dados faltantes. As colunas com mais de 300.000 valores NaN foram excluídas da análise.
# 
# - Para as outras colunas, como nós temos muito mais dados (mais de 900.000 linhas) vamos apenas exluir as linhas que contém dados NaN.
# 
# #### Treating NaN values
# 
# - Viewing the data, we realized that there is a great disparity in missing data. Columns with more than 300,000 NaN values ​​were excluded from the analysis.
# 
# - For the other columns, as we have much more data (more than 900,000 rows) we will only delete the rows that contain NaN data.

# %%
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

print(base_airbnb.isnull().sum())

# %%
base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())


# %% [markdown]
# #### Verificando os tipos de dados em cada coluna.
# 
# #### Checking data types in each column.
# 

# %%
print(base_airbnb.dtypes)
print("-"*60)
print(base_airbnb.iloc[0])

# %% [markdown]
# ##### Como as colunas de Price e Extra People estão sendo reconhecidas como Object (texto) temos que mudar tipo delas para númerico (int ou float)
# 
# ##### As the Price and Extra People columns are being recognized as Object (text) we have to change their type to numeric (int or float)
# 

# %%
# Price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)

# Extra People
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)

# %% [markdown]
# #### Vamos converter os tipos float64 e int64 para float32 e int32. 
# 
# - Memória: int32 e float32 ocupam menos espaço na memória em comparação com int64 e float64, respectivamente. Um valor int64 ou float64 ocupa 8 bytes, enquanto um valor int32 ou float32 ocupa 4 bytes. Isso pode resultar em uma redução significativa no uso de memória, especialmente em grandes DataFrames.
# 
# - Processamento mais rápido: Menos memória usada pode levar a uma performance melhorada, pois mais dados podem ser carregados na memória cache do processador, resultando em operações mais rápidas.
# 
# 
# #### Let's convert the types float64 and int64 to float32 and int32. 
# 
# - Memory: int32 and float32 take up less space in memory compared to int64 and float64 respectively. An int64 or float64 value takes up 8 bytes, while an int32 or float32 value takes up 4 bytes. This can result in a significant reduction in memory usage, especially on large DataFrames.
# 
# - Faster processing: Less memory used can lead to improved performance as more data can be loaded into the processor's cache memory, resulting in faster operations.

# %%
# Converter colunas que são float64 para float32
# Convert columns that are float64 to float32
cols_to_convert = ['host_listings_count', 'latitude', 'longitude', 'bathrooms', 'bedrooms', 'beds']
base_airbnb[cols_to_convert] = base_airbnb[cols_to_convert].astype('float32')

print("\nTipos de dados após a conversão:")
print(base_airbnb.dtypes)

# %%
# Converter colunas que são int64 para int32
# Convert columns that are int64 to int32
cols_to_convert_2 = ['accommodates', 'ano', 'mes', 'number_of_reviews', 'maximum_nights', 'minimum_nights', 'guests_included']
base_airbnb[cols_to_convert_2] = base_airbnb[cols_to_convert_2].astype('int32')

print("\nTipos de dados após a conversão:")
print(base_airbnb.dtypes)

# %% [markdown]
# ### Análise Exploratória e Tratar Outliers
# 
# - Vamos basicamente olhar feature por feature para:
#     1. Ver a correlação entre as features e decidir se manteremos todas as features que temos.
#     2. Excluir outliers (usaremos como regra, valores abaixo de Q1 - 1.5xAmplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir
#     
# - Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.
# 
# - Depois vamos analisar as colunas de valores numéricos discretos.
# 
# - Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.
# 
# Não podemos sair excluindo direto outliers, precisamos pensar exatamente no que estamos fazendo. Se não tem um motivo claro para remover o outlier, talvez não seja necessário e pode ser prejudicial para a generalização. Então precisamos balancear isso.
# 
# Ex de análise: Se o objetivo é ajudar a precificar um imóvel que você está querendo disponibilizar, excluir outliers em host_listings_count pode fazer sentido. Agora, se você é uma empresa com uma série de propriedades e quer comparar com outras empresas do tipo também e se posicionar dessa forma, talvez excluir quem tem acima de 6 propriedades tire isso do seu modelo.

# %% [markdown]
# ### Exploratory Analysis and Treating Outliers
# 
# - Let's basically look feature by feature to:
#     1. See the correlation between features and decide whether to keep all the features we have.
#     2. Exclude outliers (as a rule we will use values ​​below Q1 - 1.5xAmplitude and values ​​above Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
#     3. Confirm whether all the features we have really make sense for our model or whether any of them will not help us and whether we should exclude
#     
# - Let's start with the price columns (final result we want) and extra_people (also monetary value). These are continuous numeric values.
# 
# - Then we will analyze the columns of discrete numerical values.
# 
# - Finally, we will evaluate the text columns and define which categories make sense to keep or not.
# 
# We cannot directly exclude outliers, we need to think exactly what we are doing. If you don't have a clear reason to remove the outlier, it may not be necessary and may be detrimental to generalization. So we need to balance that.
# 
# Analysis ex: If the goal is to help price a property you are wanting to make available, excluding outliers in host_listings_count may make sense. Now, if you are a company with a series of properties and want to compare with other companies of the type and position yourself in that way, perhaps excluding those with more than 6 properties will remove this from your model.

# %%
plt.figure(figsize=(15,10))
sns.heatmap(base_airbnb.corr(numeric_only=True), annot=True, cmap='Blues')

# %% [markdown]
# #### Definição de Funções para Análise de Outliers
# 
# #### Definition of Functions for Outlier Analysis
# 

# %%
def calcular_limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = stats.iqr(coluna)
    limite_inferior = q1 - 1.5 * amplitude
    limite_superior = q3 + 1.5 * amplitude
    return limite_inferior, limite_superior


def diagrama_caixa(coluna):
    fig, (ax1, ax2)= plt.subplots(1,2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(calcular_limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)


def histograma(coluna):
    plt.figure(figsize=(15,5))
    sns.histplot(data=base_airbnb, x=coluna, element='bars')    


def delete_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = calcular_limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]    
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


def grafico_barras(coluna):
    plt.figure(figsize=(15, 5))
    # Gerando uma paleta de cores
    palette = sns.color_palette("dark", len(coluna.value_counts()))
    # Criando o gráfico de barras
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts(), palette=palette)
    ax.set_xlim(calcular_limites(coluna))
    plt.show()    

# %% [markdown]
# #### Price Column

# %%
diagrama_caixa(base_airbnb['price'])

# %%
histograma(base_airbnb['price'])

# %% [markdown]
# - Como estou construindo um modelo para imóveis comuns, acredito que os valores a cima do limite superior serão apenas de imóveis de alto padrão, que não é o meu objetivo principal. Por esse motivo, eu irei separar esses valores para treinar outro modelo apenas com esses imóveis de alto padrão.
# 
# - As I am building a model for common properties, I believe that values ​​above the upper limit will only be for high-end properties, which is not my main objective. For this reason, I will separate these values ​​to train another model only with these high-end properties.
# 

# %%
limite_inferior, limite_superior = calcular_limites(base_airbnb['price'])

outliers = base_airbnb[(base_airbnb['price'] < limite_inferior) | (base_airbnb['price'] > limite_superior)]
sem_outliers = base_airbnb[(base_airbnb['price'] >= limite_inferior) & (base_airbnb['price'] <= limite_superior)]

display(outliers)

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'price')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### extra_people

# %%
diagrama_caixa(base_airbnb['extra_people'])

# %%
histograma(base_airbnb['extra_people'])

# %%
limite_inferior, limite_superior = calcular_limites(base_airbnb['price'])

outliers = base_airbnb[(base_airbnb['price'] < limite_inferior) | (base_airbnb['price'] > limite_superior)]
sem_outliers = base_airbnb[(base_airbnb['price'] >= limite_inferior) & (base_airbnb['price'] <= limite_superior)]

display(outliers)

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'extra_people')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### host_listings_count 

# %%
diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barras(base_airbnb['host_listings_count'])

# %% [markdown]
# ##### Podemos excluir os outliers, porque para o objetivo do meu projeto, hosts com mais de 6 imóveis no airbnb não é o publico alvo do objetivo do projeto (pressupunha-se que sejam imobiliárias ou profissionais que gerenciam imóveis no airbnb).
# 
# 
# ##### We can exclude the outliers, because for the purpose of my project, hosts with more than 6 properties on airbnb are not the target audience for the purpose of the project (it was assumed that they are real estate agencies or professionals who manage properties on airbnb).
# 

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'host_listings_count')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### accommodates

# %%
diagrama_caixa(base_airbnb['accommodates'])
grafico_barras(base_airbnb['accommodates'])

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'accommodates')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### bathrooms

# %%
diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'bathrooms')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### bedrooms

# %%
diagrama_caixa(base_airbnb['bedrooms'])
grafico_barras(base_airbnb['bedrooms'])

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'bedrooms')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### beds

# %%
diagrama_caixa(base_airbnb['beds'])
grafico_barras(base_airbnb['beds'])

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'beds')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### guests_included

# %%
#diagrama_caixa(base_airbnb['guests_included'])
#grafico_barras(base_airbnb['guests_included'])
print(calcular_limites(base_airbnb['guests_included']))
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())

# %% [markdown]
# ##### Irei remover essa feature da análise. Parece que os usuários do airbnb, usam muito o valor padrão do airbnb como 1 guest included. Isso pode levar o meu modelo a considerar uma feature que não é essencial para a definição do preço, por esse motivo, me parece sensato excluir da análise.
# 
# ##### I will remove this feature from the analysis. It seems that airbnb users use the airbnb default value of 1 guest included a lot. This could lead my model to consider a feature that is not essential for defining the price, for this reason, it seems sensible to exclude it from the analysis.
# 

# %%
base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape

# %% [markdown]
# #### minimun_nights

# %%
diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barras(base_airbnb['minimum_nights'])

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'minimum_nights')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### maximum_nights

# %%
diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barras(base_airbnb['maximum_nights'])

# %%
base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape

# %% [markdown]
# #### number_of_reviews

# %%
diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barras(base_airbnb['number_of_reviews'])

# %%
base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape

# %% [markdown]
# #### Tratamento de Colunas de Valores de Texto
# #### Handling Text Value Columns
# 

# %% [markdown]
# ##### property_type

# %%
plt.figure(figsize=(15,5))
sns.set_theme(style='whitegrid')
grafico = sns.countplot(x=base_airbnb['property_type'])
grafico.tick_params(axis='x', rotation=90)

# %%
tabela_casas = base_airbnb['property_type'].value_counts()
cols_agrupp = []

for tipo in tabela_casas.index:
    if tabela_casas[tipo] < 2000:
        cols_agrupp.append(tipo)
      
for tipo in cols_agrupp:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'


print(base_airbnb['property_type'].value_counts())    

# %% [markdown]
# #### room_type

# %%
print(base_airbnb['room_type'].value_counts())    
plt.figure(figsize=(15,5))
sns.set_theme(style='whitegrid')
grafico = sns.countplot(x=base_airbnb['room_type'])
grafico.tick_params(axis='x', rotation=90)

# %% [markdown]
# #### bed_type

# %%
print(base_airbnb['bed_type'].value_counts())    
plt.figure(figsize=(15,5))
sns.set_theme(style='whitegrid')
grafico = sns.countplot(x=base_airbnb['bed_type'])
grafico.tick_params(axis='x', rotation=90)

# %%
tabela_camas = base_airbnb['bed_type'].value_counts()
colunas_agrupar_camas = []

for tipo in tabela_camas.index:
    if tabela_camas[tipo] < 10000:
        colunas_agrupar_camas.append(tipo)
      
for tipo in colunas_agrupar_camas:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'


print(base_airbnb['bed_type'].value_counts())    

# %%
print(base_airbnb['bed_type'].value_counts())    
plt.figure(figsize=(15,5))
sns.set_theme(style='whitegrid')
grafico = sns.countplot(x=base_airbnb['bed_type'])
grafico.tick_params(axis='x', rotation=90)

# %% [markdown]
# #### cancellation_policy

# %%
print(base_airbnb['cancellation_policy'].value_counts())    
cores = sns.color_palette("husl",len(base_airbnb['cancellation_policy'].unique()))
plt.figure(figsize=(15,5))
sns.set_theme(style='whitegrid')
grafico = sns.countplot(x=base_airbnb['cancellation_policy'], palette=cores)
grafico.tick_params(axis='x', rotation=90)

# %%
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
      
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'


print(base_airbnb['cancellation_policy'].value_counts())    

# %% [markdown]
# #### amenities

# %% [markdown]
# #### Como temos uma diversidade muito grande de amenities e, ás vezes, as mesmas amenities podem ser escritas de formas diferentes, irei avaliar a QUANTIDADE de amenities como parâmetro para o nosso modelo.
# 
# #### As we have a very large diversity of amenities and, sometimes, the same amenities can be written in different ways, I will evaluate the QUANTITY of amenities as a parameter for our model.
# 

# %%
print(base_airbnb['amenities'].iloc[0])

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)

# %%
base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape

# %%
diagrama_caixa(base_airbnb['n_amenities'])
grafico_barras(base_airbnb['n_amenities'])

# %%
base_airbnb, linhas_removidas = delete_outliers(base_airbnb, 'n_amenities')
print('{} Linhas Removidas'.format(linhas_removidas))

# %% [markdown]
# #### Visualização de Mapa das Propriedades
# #### Property Map View
# 

# %%
amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
            center=centro_mapa, zoom=10,
            mapbox_style='stamen-terrain')
mapa.update_layout(mapbox_style="open-street-map")
mapa.show()

# %% [markdown]
# #### encoding
# 
# - Ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true ou false e etc...)
# - Adjust the features to facilitate the work of the future model (category features, true or false, etc...)
# 
# 
#     - Features de valores True ou False, vamos substituir True por 1 e False por 0.
#     - Features de categoria (features em que os valores da coluna são texto). Irei utilizar o método de encoding de variáveis dumming.
# 
# 
#     - Category features (features where the column values ​​are text). I will use the dumming variable encoding method.
#     - Features of True or False values, we will replace True with 1 and False with 0.
# 
#     

# %%
cols_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']

base_airbnb_cod = base_airbnb.copy()

for coluna in cols_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0

# %%
cols_categ = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=cols_categ)

# %% [markdown]
# #### Modelo de Previsão
# #### Forecast Model
# 

# %% [markdown]
# - Métricas de Avaliação
# - Assessment Metrics
# 

# %%
def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'

# %% [markdown]
# - Escolha dos modelos a serem testados
# - Choosing the models to be tested
# 
#     1. RandomForest
#     2. LinearRegression
#     3. ExtraTrees

# %%
model_rf = RandomForestRegressor()
model_lr = LinearRegression()
model_et = ExtraTreesRegressor()

models = {'RandomForest': model_rf,
          'LinearRegression': model_lr,
          'ExtraTrees': model_et,
}


y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

# %% [markdown]
# - Separar os dados em Treino e Teste + Treino do Modelo
# - Separate data into Training and Testing + Model Training
# 

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10)

for nome_modelo, modelo in models.items():
    modelo.fit(X_train, y_train)
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# %%
for nome_modelo, modelo in models.items():
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# %%
pred_df=pd.DataFrame({'Valor Autal':y_test,'Valor Previsto':previsao,'Diferença':y_test-previsao})
pred_df

# %% [markdown]
# - Modelo Escolhido: ExtraTreesRegressor
# 
# Esse foi o modelo com maior valor de R² e ao mesmo tempo o menor valor de RSME. Como não tivemos uma grande diferença de velocidade e de previsão desse modelo com o modelo de RandomForest (que teve resultados semelhantes de R² e RSME), irei optar pelo ExtraTrees
# 
# O modelo de Regressão Linear obteve um resultado péssimos, com valores de R² e RSME muito piores que os outros dois.
# 
# 
# - Chosen Model: ExtraTreesRegressor
# 
# This was the model with the highest R² value and at the same time the lowest RSME value. As we did not have a big difference in speed and prediction of this model with the RandomForest model (which had similar results for R² and RSME), I will opt for ExtraTrees
# 
# The Linear Regression model obtained terrible results, with R² and RSME values ​​much worse than the other two.

# %% [markdown]
# #### Ajustes e Melhorias
# #### Adjustments and Improvements
# 

# %%
importancia_features = pd.DataFrame(model_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)

plt.figure(figsize=(15,5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)

# %% [markdown]
# - is_business_travel_ready parece não ter impacto algum no modelo. Por isso, irei excluir essas feature.
# - is_business_travel_ready appears to have no impact on some model. So I'm going to delete this feature.
# 

# %%
base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10)

model_et.fit(X_train, y_train)
previsao = model_et.predict(X_test)
print(avaliar_modelo("ExtraTrees", y_test, previsao))


# %%
base_teste = base_airbnb_cod.copy()

for coluna in base_teste:
    if 'bed_type' in coluna:
        base_teste = base_teste.drop(coluna, axis=1)

y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10)

model_et.fit(X_train, y_train)
previsao = model_et.predict(X_test)
print(avaliar_modelo("ExtraTrees", y_test, previsao))


# %% [markdown]
# #### Deploy do Projeto
# 
#     - Passo 1 -> Criar arquivo do modelo (joblib)
#     - Passo 2 -> Escolher a forma de deploy
# 
# 
# Irei optar por fazer o deploy usando o streamlit.
# 

# %%
X['price'] = y
X.to_csv('dados.csv')

# %%
import joblib

joblib.dump(model_et, 'modelo_et.joblib')

# %%
print(base_teste.columns)


