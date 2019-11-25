#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:06:51 2019

@author: Farhad
"""

import pendulum
import re
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
#from textblob import TextBlob
import spacy
nlp=spacy.load("en")


import sys
import psutil
import os
import multiprocessing
import pandas as pd
from farhad.preTexteditor import editor
from numba import jit
import string
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from farhad.time_estimate import EstimateFaster
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from farhad.Farhadcolor import bcolors, tcolors


def __info_Tweets_preprocesiing__():
    print(bcolors.GREEN)
    print("------------------------------------------------------------------------")
    print("Class:  Tweets_preprocesiing ")
    print("")
    print("input: df")
    print("Function:")
    print("          CT = Tweets_preprocesiing()")
    print("          data1 = CT.Cleaner(df)")
    print("          new_data = Remove_stop_words(data1)")
    print("          newdf = save_clean_tweets(df, target='text', date='create_at', label=None, save_file)")
    print("")
    print(" ")
    print("------------------------------------------------------------------------")
    print(bcolors.ENDC)
class Tweets_preprocesiing():
    def __init__(self,save_between=False):
        self.save_between = save_between 
        #self.df = df
        #self.new_list = []
    def save_clean_tweets(self,df,new_data,created_at='created_at',label=None,name="clean.csv"):
        
        new_df = pd.DataFrame()
        new_df[created_at] = df[created_at]
        new_df['text'] = new_data
        if label != None:
            new_df[label] = df[label]
        new_df.to_csv(name,index=False)
        
        return new_df
        
    def Remove_stop_words(self,df,title='Remove stop_words'):
        print(bcolors.BLUE)
        stop_words = [x for x in stopwords.words('english') if x!='not'] 
        
        new_list2=[]
        for num,text in enumerate(df):
            filtered = []
            text_token = word_tokenize(str(text))
            for w in text_token:
                if w not in stop_words:
                    filtered.append(w)
            new_list2.append(" ".join(filtered).strip())
            EstimateFaster(num,len(df),title)

        
        print(tcolors.RED," *** Done! ***")   
        print(bcolors.ENDC)   
        return new_list2

    def TwitterCleaner(self,text):
        stem = SnowballStemmer('english')
        #Ps = PorterStemmer('english')
        #nlp = spacy.load("en")
        
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9_]+'
        pat2 = r'https?://[^ ]+'
        www_pat = r'www.[^ ]+'
        remove_between_square_brackets = '\[[^]]*\]' #Removing the square brackets
        #part3 = string.punctuation # remove 's
        pat4 = r'@[\s]+'
        combined_pat = r'|'.join((pat1, pat2))

        negations_dic = {"isn't":"is not", "aren't":"are not", 
                         "wasn't":"was not", "weren't":"were not",
                         "haven't":"have not","hasn't":"has not",
                         "hadn't":"had not","won't":"will not",
                         "wouldn't":"would not", "don't":"do not",
                         "doesn't":"does not","didn't":"did not",
                         "can't":"can not","couldn't":"could not",
                         "shouldn't":"should not","mightn't":"might not",
                         "mustn't":"must not","isnt":"is not", "arent":"are not", 
                         "wasnt":"was not", "werent":"were not",
                         "havent":"have not","hasnt":"has not",
                         "hadnt":"had not","wont":"will not",
                         "wouldnt":"would not", "dont":"do not",
                         "doesnt":"does not","didnt":"did not",
                         "cant":"can not","couldnt":"could not",
                         "shouldnt":"should not","mightnt":"might not",
                         "mustnt":"must not","ist":"is not", "aret":"are not", 
                          
                         "havet":"have not","hasnt":"has not",
                         "hadnt":"had not","wont":"will not",
                         "wouldt":"would not", "dont":"do not",
                         "doest":"does not","didt":"did not",
                         "cant":"can not","couldnt":"could not",
                         "shouldt":"should not"}
        neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
        
        text = re.sub(combined_pat,'',text)
        soup = BeautifulSoup(text, 'lxml') #Removing the html strips
        souped = soup.get_text()
        
        try:
            bom_removed = souped.decode("utf-8").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
            
        lower_case = bom_removed.lower()
        #lower_case = stem.stem(lower_case)
        lower_case  = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case )
        stripped = re.sub(www_pat, '', lower_case)
        stripped = re.sub(pat4,'',stripped)
        stripped = re.sub(combined_pat, '', stripped)
        stripped = re.sub('https ', '', stripped)
        stripped = re.sub('rt ', '', stripped)
        stripped = re.sub("[^a-zA-Z]", " ", stripped) # only letter
        
        
        words = [stem.stem(word) for word in tok.tokenize(stripped)]
        words_new = []
        for w in words:
            if len(str(w))>1:
                words_new.append(w)
        words = words_new


        #words = [token.lemma_ for token in nlp(stripped)]
        #words = [Ps.stem(word) for word in tok.tokenize(stripped)]
        
        sentiment = (" ".join(words)).strip()
        #sentiment = TextBlob(sentiment).correct()
        
        #neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], stripped)
        #stripped = re.sub("[^a-zA-Z]", " ", stemmed_words)
        
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        #words = [x for x  in tok.tokenize(stemmed_words) if len(x) > 1]
        if len(words)>2:
            return sentiment
        else:
            return np.NaN
    
    def df_Cleaner(self,df,target='text', date='create_at', label=None, save_file='clean.csv'):
        new_list=[]
        print(bcolors.BLUE)
        for num,text in enumerate(df[target]):
            new_list.append(self.TwitterCleaner(str(text)))
            EstimateFaster(num,df[target],'clean tweets')
            
            if num%200 == 0 and self.save_between==True:
                new_df = self.save_clean_tweets(df[:len(new_list)],
                                                   new_list,created_at=date,
                                                   label=label,name=save_file)
                EstimateFaster(num,df[target],'clean tweets_save file')
        
        print(" *** Done! ***")
        print(bcolors.ENDC)
        return new_list
    