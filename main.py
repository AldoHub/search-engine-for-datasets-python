import txtai
import numpy as np
import pandas as pd

#build a small sample dataset and test it/export it for the other script to use it

np.random.seed(1)

#load the dataset
df = pd.read_csv('train.csv')
#check the columns
#print(df)

#just get 100k items from the dataset
titles = df.dropna().sample(100).TITLE.values
ids = df.dropna().sample(100).PRODUCT_ID.values

#embedding model
embeddings = txtai.Embeddings({
    'path': 'sentence-transformers/all-MiniLM-L6-v2'
})

#embed the titles - dataset
embeddings.index(titles)

#saving the embeddings
embeddings.save('embeddings.tar.gz')

result = embeddings.search('protector for cam', 5)
#actual_results = [titles[x[0]] for x in result]
actual_results = [f'Title: {titles[x[0]]}, ID: {ids[x[0]]}' for x in result]


print(actual_results)
print("RUNNING MAIN.PY")