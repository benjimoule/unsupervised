# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:12:56 2020

@author: UT2V6M
"""
#pip install matplotlib
#pip install pandas
#pip install wordcloud
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

pd.options.mode.chained_assignment = None
    
def get_wordclould(base, method = "NMF", start = 0, end = 4000, genre = "", n_top_words = 15, n_topics = 8, n_samples = 10000, n_features = 9000):
    
    if genre!= "":
        base = base[ base["genre"]== genre ]    
    base = base[ base['year'] >= start]
    base = base[ base['year'] <= end]
    
    base.reset_index(inplace = True)#compter les mots par musique 
    base['Wordcount'] = base['lyrics'].str.split().str.len()
    base['MostCommon'] = 0#construire pour chaque item la liste des mots triés par occurrence
    import collections as coll
    for i in range(base['lyrics'].size): 
        counts = coll.Counter(base['lyrics'].loc[i].split())
        base['MostCommon'].loc[i] = counts.most_common()#partie II
    
    vectorizer = text.CountVectorizer(max_df=0.95, max_features=n_features, stop_words='english')
    counts = vectorizer.fit_transform(dataset[:n_samples])
    tfidf = text.TfidfTransformer().fit_transform(counts)    
    
    # Model
    if method == "LDA":
        model = decomposition.LatentDirichletAllocation(n_components=n_topics).fit(tfidf)
    elif method == "NMF":
        model = decomposition.NMF(n_components=n_topics).fit(tfidf)
    elif method == "DBSCAN":
        model = DBSCAN(eps=0.3, min_samples=n_top_words)
        model.fit(tfidf) # check if tfidf ot counts
    else:
        model = KMeans(n_clusters=n_topics, init='k-means++', max_iter=100, n_init=1)
        model.fit(tfidf) # check if tfidf ot counts
    
    feature_names = vectorizer.get_feature_names()    
    
    if  method == "LDA" or method == "NMF":
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]]))   
            print()
        #wordcloud = WordCloud(width = 800, height = 800, 
        #            background_color ='white', 
        #            stopwords = stopwords, 
        #            min_font_size = 10).generate(comment_words) 
      
        # plot the WordCloud image                        
        #plt.figure(figsize = (8, 8), facecolor = None) 
        #plt.imshow(wordcloud) 
        #plt.axis("off") 
        #plt.tight_layout(pad = 0)        
            
    elif method == "DBSCAN":
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        labels1 = model.labels_
        print(labels1)
        n_clusters_ = len(set(labels1)) - (1 if -1 in labels1 else 0) # Number of clusters in labels
        print('Estimated number of clusters: %d' % n_clusters_)
        print()
        clusters1 = {}
        for c, i in enumerate(labels1):
            if i in clusters1:
                clusters1[i].append( tfidf[c] )
            else:
                clusters1[i] = [tfidf[c]]
        for c in clusters1:
            print(clusters1[c])
            print()
        #for topic_idx in range(n_topics):
        #    print("Topic %d:" % topic_idx)
        #    print(" ".join([feature_names[i]
        #    for i in order_centroids[9]])) 
        #    print()
    elif method == "HIERARCHICAL":
        print("coucou")
        tfidf = tfidf.toarray()
        names3 = range(len(tfidf))
        cos_tfidf = 1 - cosine_similarity(tfidf)
        linkage_matrix3 = ward(cos_tfidf)
        print("coucou2")
        dendrogram(linkage_matrix3, color_threshold=0.6*max(linkage_matrix3[:,2]), orientation="right", labels=names3)
        print("coucou3")
        plt.tight_layout()
        plt.show()
    else: #KMEANS
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        for topic_idx in range(n_topics):
            print("Topic %d:" % topic_idx)
            print(" ".join([feature_names[i]
            for i in order_centroids[topic_idx,:n_top_words]])) 
            print()

# database   
base = pd.read_csv("lyrics.csv") 
base[base['lyrics'].isnull()]
base.dropna(subset=['lyrics'], inplace = True)
base = base.replace({'\n': ' '}, regex=True)

# requests
#get_wordclould(base, method = "NMF", genre ="Rock", start = 2000 , end = 2000, n_top_words = 15, n_topics = 8, n_samples = 10000, n_features = 9000)
#get_wordclould(base, method = "LDA", genre ="Rock", start = 2000 , end = 2000, n_top_words = 15, n_topics = 8, n_samples = 10000, n_features = 9000)

get_wordclould(base, method = "HIERARCHICAL", genre ="Rock", start = 2000 , end = 2000, n_top_words = 15, n_topics = 8, n_samples = 10000, n_features = 9000)
print("end")
