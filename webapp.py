import streamlit as st
import pandas as pd
import numpy as np
from numpy import int64

import requests
import IPython.display as disp
import sklearn
from sklearn.decomposition import TruncatedSVD
import time

from streamlit.caching import suppress_cached_st_function_warning

st.markdown(
"""
<style>
.reportview-container{
        background: linear-gradient(#083513 0%,#000);
    }
</style>
"""
,unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: #1db945;'>Get Songs Recommended To Your Choice</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #BEFFCF ; font-style: italic'> <b> Know Your Vibe.</b></h2>", unsafe_allow_html=True)

songs_df= pd.read_csv('1mil-songs.csv')

song_user_count= pd.read_table('triplets.txt')
song_user_count.columns= ['user_id','song_id','listen_count']

merged_df= pd.merge(song_user_count, songs_df, on='song_id')

pt_df= merged_df.pivot_table(values='listen_count',index='user_id',columns='title',fill_value=0)

transpose= pt_df.values.T

SVD= TruncatedSVD(n_components= 20, random_state= 17)
result_matrix= SVD.fit_transform(transpose)

corelation_m= np.corrcoef(result_matrix)

songnames= pt_df.columns
songlist= list (songnames)

inp = st.text_input("Enter your song")

st.cache()
if st.button("Recommend"):
        
    if (inp in songlist):
        my_bar= st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete+1)
            suppress_cached_st_function_warning = True
        st.write("*Here are the songs recommended for you:* :sunglasses:")
        recsong = corelation_m[songlist.index(inp)]
        x=(list(songnames[(recsong <1.0) & (recsong >0.9)]))
        st.table(x)
    else:
        st.error("Sorry, this song isn't on our list :broken_heart: Maybe try another?")
