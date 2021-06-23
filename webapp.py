import streamlit as st
import pandas as pd
import numpy as np
from numpy import int64

import requests
import IPython.display as disp
import sklearn
from sklearn.decomposition import TruncatedSVD

st.header("Welcome To Our")
st.title("Genre-Based Music Recommendation System")
songs_df= pd.read_csv('1mil-songs.csv')
songs_df2= songs_df.loc[1:50000]
song_user_count= pd.read_table('triplets.txt')
song_user_count.columns= ['user_id','song_id','listen_count']
merged_df= pd.merge(song_user_count, songs_df2, on='song_id')

pt_df= merged_df.pivot_table(values='listen_count',index='user_id',columns='title',fill_value=0)

transpose= pt_df.values.T

SVD= TruncatedSVD(n_components= 20, random_state= 17)
result_matrix= SVD.fit_transform(transpose)

corelation_m= np.corrcoef(result_matrix)

songnames= pt_df.columns
songlist= list (songnames)

st.header("Hey there!")
st.subheader("Need some music recommendations based on songs you already love?")
st.subheader("Enter a song below & we'll try our best to recommend you similar songs!")
inp = st.text_input("")

if st.button("Recommend"):
    if (inp in songlist):
        st.write("*Here are the songs recommended for you:*")
        recsong = corelation_m[songlist.index(inp)]
        st.write(list(songnames[(recsong <1.0) & (recsong >0.9)]))
        
    else:
        st.write("Sorry :( The song you entered is not in our database! Maybe try another?")

    st.subheader("Hope you enjoy these songs!")
