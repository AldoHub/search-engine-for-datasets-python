import txtai
import numpy as np
import pandas as pd
import streamlit as st

def load_data_and_embeddings():
    np.random.seed(1)
    df = pd.read_csv('train.csv')
    #use a sample since is a really big dataset
    titles = df.dropna().sample(100).TITLE.values
    
    embeddings = txtai.Embeddings(
        path='sentence-transformers/all-MiniLM-L6-v2'
    )

    #load the embeddings we saved on the other script
    embeddings.load('embeddings.tar.gz')
    return titles, embeddings


#streamlit
titles, embeddings = st.cache_data(load_data_and_embeddings)()

st.title('Amazon Items Search')
query = st.text_input('Enter a search case', '')

#https://neuml.github.io/txtai/embeddings/
if st.button('Search'):
    if query:
        st.write('Returning results...')  
        result = embeddings.search(query, 5)
        
        actual_results = [titles[x[0]] for x in result]
        
        for res in actual_results:
            st.write(res)
           
    else:
        st.write('Please Enter a search case')        

 #streamlit run <filename.py> OR python -m streamlit run <filename.py>-- in order to run this app       
 