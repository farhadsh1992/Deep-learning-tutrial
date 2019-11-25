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


class cleaning():
    
    
    
    def __init__(self):
        import re
        from nltk.tokenize import WordPunctTokenizer
        from bs4 import BeautifulSoup
        from textblob import TextBlob
        
    
    def __version__():
        print("1.2.3")
        print("This Pakage designe by Farhad")
        print("github: Farhadsh1992")
        print("linkden: www.linkedin.com/in/farhad-shadmand-a86170121")
        
    def help():
        return 
        
    def TwitterCleaner(self,text):
        
        
        
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9_]+'
        pat2 = r'https?://[^ ]+'
        www_pat = r'www.[^ ]+'
    
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
        stripped = re.sub(www_pat, '', lower_case)
        stripped = re.sub(combined_pat, '', stripped)
        
        
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], stripped)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
        
        return (" ".join(words)).strip()
    
    def ckeckspeller(self,text):
        
        result = TextBlob(str(text)).correct()
        
        return result
    
    
    
class cleaning_DataFrame():  
    
    
    def __init__(self):
        
        self.ListC = []
    
    
    def ckeckspeller(self,df):
        
        start = pendulum.now()
        aa=0
        for num,li in enumerate(df):
            
            lA = TextBlob(str(li)).correct()
            self.ListC.append(lA)
        if (num%5000)==1:
            stop = pendulum.now()
            dift1 = (start-stop)*1
            estimatTime = -(dift1*len(df)/num)
            aa+=1
            run = (str(num)+'/'+str(len(df))+
                   " _ spended time:" +str(dift1) +" _ running "+ "."*aa +"#")
            sys.stdout.write('\r'+ run + "Estimate to finish:" + str(estimatTime))
        return self.ListC
      
    