#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:30:08 2019

@author: Farhad
"""
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
import pyLDAvis.gensim
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import numpy as np

def bag_of_words_for_unigrams(x):
    x = word_tokenize(str(x))
    return x

def bag_of_words_for_bigrams(x):
    x = word_tokenize(str(x))
    texts = []
    for i in range(len(x)-1):
        
        texts.append('{} {}'.format(x[i],x[i+1]))
    return texts
def make_dictionary(list_dic, orginal_list):
    bigram = gensim.models.Phrases(orginal_list)
    bigram_dic = gensim.models.Phrases(list_dic)
    
    texts =  [bigram[line] for line in orginal_list]
    texts_dic =  [bigram_dic[line] for line in list_dic]
    dictionary = Dictionary(texts_dic)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return corpus, dictionary

def evaluate_graph_for_max_label(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    warnings.filterwarnings("ignore")
    for num_topics in range(2, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(2, limit)
    m = (i for i in range(len(c_v)) if c_v[i]==max([c_v]))
    print(max(c_v),m)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    
def find_probility_for_label(ldamodel, corpus):
    
    """
    This finctuion extract probability for every row according to threshold
    
    Parameters:
    ----------
    ldamodel : model of LDA
    corpus : Gensim corpus
    
    
    Returns:
    -------
    probabilitylist : probabilitylist list for every row 
    leballist : label for ever row
    """
    lda_corpus = ldamodel[corpus]
    
    #find threshold
    scores = list(chain(*[[score for topic_id,score in topic] \
                          for topic in [doc for doc in lda_corpus]]))
    threshold = sum(scores)/len(scores)
    print('threshold for probability:',threshold)

    lda_corpus = ldamodel[corpus]
    #extract probability
    probabilitylist =[]
    leballist = []
    for n in range(len(lda_corpus)): #len(lda_corpus)
        ilab = lda_corpus[n]
        probabilityL = [ilab[i][1] for i in range(len(ilab)) if  ilab[i][1] > threshold]
        lebal = [ ilab[i][0] for i in range(len(ilab)) if  ilab[i][1] > threshold ]
        #print(probilityL )
        #print(lebal)
        probabilitylist.append(probabilityL)
        leballist.append(lebal)
        
        
    return probabilitylist,leballist

def array_probablity_label(probilityL_list,lebal_list,num_topic):
    List_aray = []
    for labels in lebal_list:
        array = np.zeros(num_topic)
        for la in labels:
            array[la]= probilityL_list[la]
        List_aray.append(array)
    return List_aray

def arrowforlebels(probability_list,lebal_list, num_topics):
    """
    This finctuion get  problitly for evry row and compute a mount 
    probability for every class
    
    Parameters:
    ----------
   probabilitylist : probabilitylist list for every row  
    leballist : label for ever row
    
    
    Returns:
    -------
    porb : a probabilitylist for evry class (precent)
    
    
    """
    arrow_probability = []
    for num in range(len(probability_list)):
        k = np.zeros(num_topics)
        for i in range(len(probability_list[num])):
            k[lebal_list[num][i]] = probability_list[num][i]
        arrow_probability.append(k)

    listt = []
    for i in range(num_topics):
        listt.append([])
        for j in range(len(arrow_probability)):
            listt[i].append(arrow_probability[j][i])

    porb = []
    #print(listt)
    listt2 = []
    for i in listt:
        listt2.append(sum(i))
    s= sum(listt2)
    for i in range(len(listt)):
        porb.append(round(sum(listt[i])*100/s,2))
    return porb,arrow_probability