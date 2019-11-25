#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:09:42 2018

@author: Farhad

Title: Defining data cleaning function: (best version)
"""


import pendulum
import re
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
import sys
import psutil
import os
import multiprocessing
import pandas as pd
from farhad.preTexteditor import editor
from numba import jit
import string
from nltk.stem import SnowballStemmer
from farhad.time_estimate import EstimateFaster

print("""pakage that runing: pendulum,multiprocessing
      re,nltk.tokenize,bs4,textblob,nltk.tokenize,sys,psutil,os""")
class help():
    import pandas as pd
    df_cleaning = pd.DataFrame()
    df_cleaning['funcation']=['cleaning.TwitterCleaner','cleaning.ckeckspeller']
    df_cleaning['init']=['text','text']
    df_cleaning['type']=['str','str']
    df_cleaning['responsible']=["""
      clean data _ remove website address _
      lowerspace _  n't to not _ nt to not
    
    """,""" checkspell with textblob.TextBlob """]
    
    df_cleaning.head()



        
def TwitterCleaner(text):
    stem = SnowballStemmer('english')
        
    tok = WordPunctTokenizer()
    pat1 = r'@[A-Za-z0-9_]+'
    pat4 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[^ ]+'
    www_pat = r'www.[^ ]+'
    part3 = string.punctuation # remove 's
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
        
        
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
        
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
            
    lower_case = bom_removed.lower()
    lower_case = stem.stem(lower_case)
    stripped = re.sub(www_pat, '', lower_case)
    stripped = re.sub(combined_pat, '', stripped)
    stripped = re.sub('https ', '', stripped)
    stripped = re.sub('rt ', '', stripped)
    stripped = re.sub(pat4, '', stripped)
    
        
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], stripped)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
        
    return (" ".join(words)).strip()
 
    
    
    
class cleaning_DataFrame():  
    
    
    def __init__(self):
        
        self.ListC = []
    
    @jit
    def ckeckspeller(self,df,label,name):
        
        start = pendulum.now()
        aa=0
        Listlabel=[]
        for num,li in enumerate(df):
            df1 = pd.DataFrame()
            lA = TextBlob(str(li)).correct()
            self.ListC.append(str(lA))
            Listlabel.append(label[num])
            df1['text'] = pd.Series(self.ListC)
            df1['label'] = Listlabel
            df1.to_csv('temperray_clean_file{}.csv'.format(name))
            #editie = editor()
            #multiprocessing.Process(target=editie.Editor,args=(li,))
            #self.ListC.append(lA)
            
            
            EstimateFaster(df,num,'check_spell')    
        print ("%%%%% finshed %%%%")
        return self.ListC
      
class cleaning_DataFrame_without_label():  
    
    
    def __init__(self):
        
        self.ListC = []
    
    #@jit
    def ckeckspeller(self,df,name):
        
        start = pendulum.now()
        aa=0
        Listlabel=[]
        for num,li in enumerate(df):
            df1 = pd.DataFrame()
            
            lA = TextBlob(str(li)).correct()
            self.ListC.append(str(lA))
            
            df1['text'] = pd.Series(self.ListC)
            
            df1.to_csv('temperray/temperray_clean_file{}.csv'.format(name))
            #editie = editor()
            #multiprocessing.Process(target=editie.Editor,args=(li,))
            #self.ListC.append(lA)
            
            
            numb=num+1
            stop = pendulum.now()
            dift1 = abs(start-stop)*1
            estimatTime = abs(abs(dift1*len(df)/numb)-dift1)
                
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = psutil.virtual_memory().percent
            cpuUse = psutil.cpu_percent(interval=1, percpu=True)
            #aa+=1
            run = ("["+str(num)+
                     '/'+str(len(df))+"] _ [ time:" + str(dift1) +
                     "] _ ["+" Remain:" + str(estimatTime)+
                     "] _ ["+" Memory:"+str(memoryUse)+"% ] _ ["+" CPU:"+str(cpuUse)+"% ]")
            sys.stdout.write('\r'+ run)
        print ("%%%%% finshed %%%%")
        return self.ListC    