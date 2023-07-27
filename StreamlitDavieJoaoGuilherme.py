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

sns.set_theme(style="white")



st.title('Trabalho final (ainda incompleto) - Davi e João Guilherme')

st.header("Motivação do trabalho")
st.write('Vamos analisar dados relativos aos filmes no top 2000 do IMDb por rating nessa plataforma.')
st.write('Primeiramente, selecione o arquivo filmes.csv na coluna do lado esquerdo.')


df_csv = st.sidebar.file_uploader('Arquivo',
                                  type=['csv','zip'],
                                  accept_multiple_files=False,
                                  key="fileUploader")

if df_csv is not None:
  filmes = pd.read_csv(df_csv)


st.header('Alguns gráficos')

option = st.selectbox("Selecione um gráfico:", ["Heatmap de correlação",
                                               "Top 10 filmes por nota do IMDb",
                                               "Top 10 filmes por lucro bruto nos EUA",
                                               "Top 8 anos com mais lançamentos de filmes no top 2000 do IMDb",
                                               "KDE plot da duração dos filmes",
                                               "Top 10 gêneros"])

if option == "Heatmap de correlação":
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

elif option == "Top 8 anos com mais lançamentos de filmes no top 2000 do IMDb":
    top_anos = filmes['ano'].value_counts().sort_values(ascending=False)[:8]
    fig, axs = plt.subplots(figsize=(15,5))
    g = sns.barplot(x=top_anos.index,
                    y=top_anos, order=top_anos.index, palette = 'tab10')
    g.set_title("Top 8 anos com mais lançamentos de filmes no top 2000 do IMDb", weight = "bold")
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
elif option == "KDE plot da duração dos filmes":
    fig, axs = plt.subplots(figsize=(15,5))
    g = sns.kdeplot(filmes['duracao'])
    g.set_title("Duração dos filmes", weight = "bold")
    st.pyplot(fig)

elif option == "Top 10 gêneros":
    genre = []
    for x in filmes['genero']:
        for y in x.split(','):
            genre.append(y.strip().lower())
    count=Counter(genre)
    count=count.most_common()[:10]
    x,y=map(list,zip(*count))

    fig, axs=plt.subplots(figsize=(15,5))
    g=sns.barplot(data=pd.DataFrame(y, index=x, columns=['numero']),
                  y='numero', x=pd.DataFrame(y, index=x, columns=['numero']).index)
    g.set_title("Top 10 gêneros", weight = "bold")
    plt.xticks(rotation=90)
    st.pyplot(fig)



