# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:02:38 2020

@author: benjamin.policand
"""
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.preprocessing import normalize;
from sklearn.decomposition import NMF
import nltk
nltk.download('punkt')
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import en_core_web_sm
nlp = en_core_web_sm.load()

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("lyrics.csv") 
# Preview the first 5 lines of the loaded data 
print(data.head())
print(data.columns.tolist())
print(data.year.unique())

data_group_by_year=data.groupby(['year'])
print(data.groupby(['year']))
print(data.groupby(['year'])['index'].count())
#print(data_group_by_year.describe())
#Let's take 2016
data_2016=data[data['year']==2016]
print(data.columns.tolist())
print(data_2016)

data_text=nlp(data_2016['lyrics'].iloc[0])
filtered_sentence = [] 
word_tokens = word_tokenize(data_text)  
stop_words=set(stopwords.words('english')) 

for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w)
print(filtered_sentence)
vectorizer = CountVectorizer(analyzer='word', max_features=5000);
x_counts = vectorizer.fit_transform(filtered_sentence);

# Next, we set a TfIdf Transformer, and transform the counts with the model.

# In[81]:

transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);


# And now we normalize the TfIdf values to unit length for each row.

# In[82]:

xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)


# And finally, obtain a NMF model, and fit it with the sentences.

# In[84]:
num_topics=10
#obtain a NMF model.
model = NMF(n_components=num_topics, init='random');
#fit the model
model.fit(xtfidf_norm)


# In[136]:

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);


# In[139]:
#matrix = [data_text.split() for item in data_text.split('\n')[:-1]]
#print(matrix)
print('---------------model---------------------')
print(get_nmf_topics(model, 20))
#import numpy as np
#X = matrix
#from sklearn.decomposition import NMF
#model = NMF(n_components=2, init='random', random_state=0)
print('---------------W---------------------')
W = model.fit_transform(xtfidf_norm)
print(W)
print('---------------H---------------------')
H = model.components_
print(H)
