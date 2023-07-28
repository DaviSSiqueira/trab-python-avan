# Importação de bibliotecas
import numpy as np
import pandas as pd
import requests
import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sns.set_theme(style="white")



st.title('Trabalho final (ainda incompleto) - Davi e João Guilherme')

st.header("Motivação do trabalho")
st.write('Vamos analisar dados relativos aos filmes no top 1000 do IMDb por rating nessa plataforma.')
st.write('Primeiramente, selecione o arquivo filmes.csv abaixo')

df_csv = st.file_uploader('Selecione o arquivo',
                                  type=['csv'])

if df_csv:
    filmes = pd.read_csv(df_csv)
    st.dataframe(filmes.head())

    st.header('Alguns gráficos')

    option = st.selectbox("Selecione um gráfico ou grupo de gráficos:", ["Análise univariada: histogramas e gráficos de barras",
                                                   "Análise bivariada: regressões simples e boxplots",          
                                                   "Heatmap de correlação",
                                                   "Top 10 filmes por nota do IMDb",
                                                   "Top 10 filmes por lucro bruto nos EUA",
                                                   "Top 8 anos com mais lançamentos de filmes no top 2000 do IMDb",
                                                   "Top 10 gêneros"])

    if option == "Análise univariada: histogramas e gráficos de barras":
        fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 10))
        for index, column in enumerate(['ano', 'duracao', 'nota_imdb', 'metascore', 'num_votos', 'lucro_eua']):
            ax = axes.flatten()[index]
            ax.hist(filmes[column], bins=30, label = column)
            ax.legend(loc = "best")
        plt.suptitle("Histogramas", size = 20)
        st.pyplot(fig)

        top_generos = filmes['genero1'].value_counts().sort_values(ascending=False)
        fig, axs = plt.subplots(figsize=(15,5))
        g = sns.barplot(x=top_generos.index,
                        y=top_generos, order=top_generos.index, palette = 'tab10')
        g.set_title("Gráfico de barras do primeiro gênero dos filmes", weight = "bold")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        top_generos2 = filmes['genero2'].value_counts().sort_values(ascending=False)
        fig, axs = plt.subplots(figsize=(15,5))
        g = sns.barplot(x=top_generos2.index,
                        y=top_generos2, order=top_generos2.index, palette = 'tab10')
        g.set_title("Gráfico de barras do segundo gênero dos filmes", weight = "bold")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        top_generos3 = filmes['genero3'].value_counts().sort_values(ascending=False)
        fig, axs = plt.subplots(figsize=(15,5))
        g = sns.barplot(x=top_generos3.index,
                        y=top_generos3, order=top_generos3.index, palette = 'tab10')
        g.set_title("Gráfico de barras do terceiro gênero dos filmes", weight = "bold")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif option == "Análise bivariada: regressões simples e boxplots":
        f = sns.lmplot(data=filmes, x='metascore', y='nota_imdb')
        st.pyplot(f)

        f = sns.lmplot(data=filmes, x='lucro_eua', y='nota_imdb')
        st.pyplot(f)

        f = sns.lmplot(data=filmes, x='duracao', y='nota_imdb')
        st.pyplot(f)

        f = sns.lmplot(data=filmes, x='num_votos', y='nota_imdb')
        st.pyplot(f)

        fig, axs = plt.subplots(figsize=(15,5))
        f = sns.boxplot(data=filmes, x="nota_imdb", y="genero1")
        st.pyplot(fig)

        fig, axs = plt.subplots(figsize=(15,5))
        f = sns.boxplot(data=filmes, x="nota_imdb", y="genero2")
        st.pyplot(fig)

        fig, axs = plt.subplots(figsize=(15,5))
        f = sns.boxplot(data=filmes, x="nota_imdb", y="genero3")
        st.pyplot(fig)

    elif option == "Heatmap de correlação":
        filmes_num = filmes.drop(['titulo', 'genero'], axis=1)
        corr = filmes_num.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        st.pyplot(f)

    elif option == "Top 10 filmes por nota do IMDb":
        top_filmes = filmes.sort_values(['nota_imdb'], ascending = False)
        fig, axs = plt.subplots(figsize=(15,5))
        g = sns.barplot(x=top_filmes['titulo'][:8], y=top_filmes['nota_imdb'][:8], palette = 'tab10')
        g.set_title("Top 10 filmes por nota do IMDb", weight = "bold")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif option == "Top 10 filmes por lucro bruto nos EUA":
        top_filmes_lucro = filmes.sort_values(['lucro_eua'], ascending = False)
        fig, axs = plt.subplots(figsize=(15,5))
        g = sns.barplot(x=top_filmes_lucro['titulo'][:8], y=top_filmes_lucro['lucro_eua'][:8], palette = 'tab10')
        g.set_title("Top 10 filmes por lucro bruto nos EUA", weight = "bold")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif option == "Top 8 anos com mais lançamentos de filmes no top 1000 do IMDb":
        top_anos = filmes['ano'].value_counts().sort_values(ascending=False)[:8]
        fig, axs = plt.subplots(figsize=(15,5))
        g = sns.barplot(x=top_anos.index,
                        y=top_anos, order=top_anos.index, palette = 'tab10')
        g.set_title("Top 8 anos com mais lançamentos de filmes no top 1000 do IMDb", weight = "bold")
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
    st.header('Modelo de Machine Learning')
    
    st.write('Vamos utilizar um RandomForestRegressor para prever a nota do IMDb do filme com base nas seguintes features: ano, duração, metascore, número de votos (proxy de popularidade), lucro bruto nos EUA e o primeiro gênero do filme (será feito um OneHotEncoding para esta feature).')
    
    st.write('O modelo pode demorar algum tempo para rodar. O dataset de treino das features que utilizamos é o seguinte:')
    
    filmes['genero1'] = filmes['genero1'].astype("category")
    filmes['genero1new'] = filmes['genero1'].cat.codes
    enc = OneHotEncoder()
    enc_filmes = pd.DataFrame(enc.fit_transform(
        filmes[['genero1new']]).toarray())
    filmes = filmes.join(enc_filmes)
    
    X = filmes.drop(['titulo','genero','genero1','genero2','genero3','genero1new','nota_imdb'], axis=1).values
    X_columns = filmes.drop(['titulo','genero','genero1','genero2','genero3','genero1new','nota_imdb'], axis=1).columns
    y = filmes['nota_imdb']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)
    
    X_train = pd.DataFrame(X_train, columns=X_columns.astype(str))
    X_test = pd.DataFrame(X_test, columns=X_columns.astype(str))
    
    st.dataframe(X_train.head())
    
    parameters = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2,3,4,5],
    }
    regr = RandomForestRegressor(random_state=100)

    gscv = GridSearchCV(regr, parameters)
    gscv.fit(X_train, y_train)

    st.write(f'O melhor estimador encontrado pelo GridSearchCV é {gscv.best_estimator_}.')
    
    y_pred_train = gscv.predict(X_train)
    st.write(f'O erro quadrático médio para o dataset de treinamento foi {mean_squared_error(y_train, y_pred_train)}.')
    
    y_pred = gscv.predict(X_test)
    st.write(f'O erro quadrático médio para o dataset de treinamento foi {mean_squared_error(y_test, y_pred)}.')
    
    fig, axs=plt.subplots(figsize=(15,5))
    plt.plot(y_test, y_pred, 'b.')
    plt.title("Resultados", fontsize=16)
    plt.xlabel("Real", fontsize=14)
    plt.ylabel("Predito", fontsize=14)
    st.pyplot(fig)



