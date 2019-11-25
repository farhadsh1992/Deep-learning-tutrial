#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:12:27 2019

@author: Farhad
issue : get news of differnt website
"""
import requests
from bs4 import BeautifulSoup
import re
from dask import delayed
import sys
import csv
import pandas as pd
print("----------------------------------------------------------")
print('__info_getnews__(number)')
print('number = 1, Yahoo_finacial_news_extract')
print('number = 2, CENT_news_extract')
print('number = 3, New_York_Times_news_extract')
print('number = 4, marketwatch_news_extract')
print("----------------------------------------------------------")
def __info_getnews__(number):
    if number ==1:
        print('EstimateFaster')
        print("----------------------------------------------------------")
        print("*** frist Applicayion ***")
        print("class: Yahoo_finacial_news_extract(words)")
        print('Yahoo_finacial_search_topic()')
        print('Yahoo_finacial_search_subscript()')
        print('Yahoo_save_file(savename="data/Yahoo_finacial_new.csv")')
        print(" ")
        print('csv file save automate')
        print('output: self.titles_list')
        print("----------------------------------------------------------")
    if number ==2:
        print("----------------------------------------------------------")
        print("*** Secound Applicayion ***")
        print("class: CENT_news_extract(words)")
        print('CENT_news_topics()')
        print('CENT_news_search_subscript()')
        print('CENT_save(savename="data/CENT_new.csv")')
        print(" ")
        print('csv file save automate')
        print('output: self.titles_list')
        print("----------------------------------------------------------")
    if number ==3:
        print("----------------------------------------------------------")
        print("*** Secound Applicayion ***")
        print("class: Nytimes_news_extract(words)")
        print('Nytimes_news_topics()')
        print('Nytimes_news_search_subscript()')
        print('Nytimes_save(savename="data/Nytimes_new.csv")')
        print(" ")
        print('csv file save automate')
        print('output: self.titles_list')
        print("----------------------------------------------------------")
    if number ==4:
        print("----------------------------------------------------------")
        print("*** Secound Applicayion ***")
        print("class: marketwatch_news_extract(words)")
        print('marketwatch_news_topics()')
        print('marketwatch_news_search_subscript()')
        print('marketwatch_save(savename="data/marketwatch_new.csv")')
        print(" ")
        print('csv file save automate')
        print('output: self.titles_list')
        print("----------------------------------------------------------")

def EstimateFaster(num,xlist, description):
    """
    Estimate length of loop and time 
    """
    num+=1
    run = ("["+str(num)+'/'+str(len(xlist))+"]["+str(description)+']')
        
    sys.stdout.write('\r'+ run)

class Yahoo_finacial_news_extract():
    def __init__(self, words):
        self.words = [x.lower() for x in words]
        self.titles_list = {}
        self.titles_list['date'] = []
        self.titles_list['titles'] = []
        self.titles_list['url_titles'] = []
        self.titles_list['main_text']= []
        
        

    
    def Yahoo_finacial_search_topic(self):
        """
        funcation:
        search in Yahoo financial website and looking for the news that match with oure 
        worsd and give url and title list of them 
        ______________________________________
        input:
           words: (list,str) it is a list of words we looking for them in titles of news
        ______________________________________
        output:
         titles_list: (dic, str) 
         1. url of titles news that match with words list
         2. title of news that match with words list
         """
        
        topUrl = 'https://finance.yahoo.com/?guccounter=1'
        respfirst = requests.get( topUrl)
        soup = BeautifulSoup(respfirst.text, 'lxml')
        titles = soup.find_all('a', href=True)
    
        
        for word in self.words:
            for tit in titles:
                m = re.search(word,tit.text.lower())
                
                if m is not None:
                    self.titles_list['titles'].append(tit.text)
                    self.titles_list['url_titles'].append(tit['href'])
        if self.titles_list is None:
            return print("Don't have news about {}".foramt(words))
        
    def Yahoo_finacial_search_subscript(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with isuue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
        
        for num,url in enumerate(self.titles_list['url_titles']):
            
            EstimateFaster(num,self.titles_list['url_titles'], 'number of news that are found ')
            
            resp_sec = requests.get('https://finance.yahoo.com{}'.format(url))
            soup_sec = BeautifulSoup(resp_sec.text,'lxml')
            
            body = soup_sec.find('body')
            date = soup_sec.find('time',{'itemprop':"datePublished"}).text
            
            main_text = [x.text for x in body.find_all('p',{'type':'text'})]
            main_text = ' '.join(main_text[:])
            
            self.titles_list['main_text'].append(main_text)
            self.titles_list['date'].append(date)
        
    def Yahoo_save_file(self,savename='data/Yahoo_finacial_new.csv'):
        """
        funcation:
            it search in finance.yahoo.com/news/ with speacial keywords.
            it is made from two fucnation,
                1. Yahoo_finacial_search_topic
                2. Yahoo_finacial_search_subsript
        ______________________________________
        input:
            words: (list,str) it is a list of words we looking for them in titles of news
        ______________________________________
        output: 
             1. a csv file of news that it collected 
             2. give a dictounary from news (title, url, main_text)
        """
   
        
        
        
        with open (savename,'a') as f:
            for num,col in enumerate(self.titles_list['titles']):
                file = csv.writer(f)
                
                llist =[self.titles_list['date'][num],
                        str(self.titles_list['titles'][num]),
                        str(self.titles_list['url_titles'][num]),
                        str(self.titles_list['main_text'][num])]
                EstimateFaster(num,self.titles_list['url_titles'],
                 'number of news from Yahoo that are saved[scripy:{}]'.format(self.words))
                file = file.writerow(llist)
        print("*** Done! ***")
        df_yahoo = pd.DataFrame(self.titles_list)
        
        return df_yahoo
#"-----------------------------------------------------------------------------"

def EstimateFaster(num,xlist, description):
    """
    Estimate length of loop and time 
    """
    num+=1
    run = ("["+str(num)+'/'+str(len(xlist))+"]["+str(description)+']')
    sys.stdout.write('\r'+ run)



class CENT_news_extract():
    
    def __init__(self,words):
        self.words = [x.lower() for x in words]
        self.titles_list = {}
        self.titles_list['date'] = []
        self.titles_list['titles'] = []
        self.titles_list['url_titles'] = []
        self.titles_list['main_text']= []
        
   
        
    def CENT_news_topics(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with issue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
    
        
    
        topUrl = 'https://www.cnet.com/news/'
        respfirst = requests.get( topUrl)
        soup = BeautifulSoup(respfirst.text, 'lxml')
    
 
        titlesOne = soup.find_all('h3',{'class':"h"})
    
        for subtitle in titlesOne:
            titles2 = subtitle.find_all('a',href=True)
            for tit in titles2:
                for word in self.words:
                    m = re.search(word,tit.text.lower())
                    if m is not None:
                        self.titles_list['titles'].append(tit.text)
                        self.titles_list['url_titles'].append('https://www.cnet.com{}'.format(tit['href']))
            
            
            
        titlesTwo = soup.find_all('a',{'class':"assetHed"},href=True)     
        for tit in titlesTwo:
            for word in self.words:
                m = re.search(word,tit.text.lower())
                if m is not None:
                    self.titles_list['titles'].append(tit.text.lstrip())
                    self.titles_list['url_titles'].append('https://www.cnet.com{}'.format(tit['href']))
        
    def CENT_news_search_subscript(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with issue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
        
        for num,url in enumerate(self.titles_list['url_titles']):
            EstimateFaster(num,self.titles_list['url_titles'], 'number of news on CNET that are found ')
        
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            body = soup.findAll('p')
            try:
                condition = ['CNET también está disponible en español.',soup.find('p',{'class':"article-dek"}).text , soup.find('span',{'class':"credit"}).text]
            except:
                condition = ['CNET también está disponible en español.',soup.find('p',{'class':"article-dek"}) , soup.find('span',{'class':"credit"})]
            main_text = [x.text for x in body if x.text not in condition]
            main_text = ' '.join(main_text[:])
            date = soup.find('span',{'class':"formattedDate"}).text
            self.titles_list['main_text'].append(main_text)
            self.titles_list['date'].append(date)
            
            
        
    def CENT_save(self,nameSave='data/CENT_new.csv'):
        """
        funcation:
            it search in CENT News with speacial keywords.
            it is made from two fucnation,
                1. Yahoo_finacial_search_topic
                2. Yahoo_finacial_search_subsript
        ______________________________________
        input:
            words: (list,str) it is a list of words we looking for them in titles of news
        ______________________________________
        output: 
             1. a csv file of news that it collected 
             2. give a dictounary from news (title, url, main_text)
        """
   
        #titles_list1 = delayed(Yahoo_finacial_search_topic)(self.words)
        #main_text = delayed(Yahoo_finacial_search_subscript)(titles_list1)
        #self.titles_list = main_text.compute()
        with open (nameSave,'a') as f:
            for num,col in enumerate(self.titles_list['titles']):
                file = csv.writer(f)
                
                llist =[self.titles_list['date'][num],
                        str(self.titles_list['titles'][num]),
                        str(self.titles_list['url_titles'][num]),
                        str(self.titles_list['main_text'][num])]
                EstimateFaster(num,self.titles_list['url_titles'], 
                'number of news from CNET.com that are saved [scripy:{}]'.format(self.words))
                file = file.writerow(llist)
        print("*** Done! ***")
        df_CENT = pd.DataFrame(self.titles_list)
        #df_CENT.head()
        return df_CENT


#"-----------------------------------------------------------------------------"
        
class Nytimes_news_extract():
    
    def __init__(self,words):
        self.words = [x.lower() for x in words]
        self.titles_list = {}
        self.titles_list['date'] = []
        self.titles_list['titles'] = []
        self.titles_list['url_titles'] = []
        self.titles_list['main_text']= []
        
   
        
    def Nytimes_news_topics(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with issue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
    
        
    
        url = ' https://www.nytimes.com'
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'lxml')
    
 
        titles = soup.find_all('a',href=True)
        titles2= []
        for title in titles:
            titles2 = title.find_all('div')
            for tit in titles2:
                for word in self.words:
                    m = re.search(word,tit.text.lower())
                    if m is not None:
                        self.titles_list['titles'].append(tit.text)
                        self.titles_list['url_titles'].append('https://www.nytimes.com{}'.format(title['href']))
    
        
            
            
        
        
    def Nytimes_news_search_subscript(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with issue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
        
        for num,url in enumerate(self.titles_list['url_titles']):
            EstimateFaster(num,self.titles_list['url_titles'], 'number of news on New_York_Times that are found ')
        
            req = requests.get(url)
            soup_body = BeautifulSoup(req.text, 'lxml')
            date  = soup_body.find('time',{"class":"css-qddhf4"}).text
            main_text  = soup.find('div',{'class':"css-53u6y8"}).text
            
            
            self.titles_list['main_text'].append(main_text)
            self.titles_list['date'].append(date)
            
            
        
    def Nytimes_save(self,nameSave='data/Nytimes_new.csv'):
        """
        funcation:
            it search in New York Times News with speacial keywords.
            it is made from two fucnation,
                1. Yahoo_finacial_search_topic
                2. Yahoo_finacial_search_subsript
        ______________________________________
        input:
            words: (list,str) it is a list of words we looking for them in titles of news
        ______________________________________
        output: 
             1. a csv file of news that it collected 
             2. give a dictounary from news (title, url, main_text)
        """
   
        #titles_list1 = delayed(Yahoo_finacial_search_topic)(self.words)
        #main_text = delayed(Yahoo_finacial_search_subscript)(titles_list1)
        #self.titles_list = main_text.compute()
        with open (nameSave,'a') as f:
            for num,col in enumerate(self.titles_list['titles']):
                file = csv.writer(f)
                
                llist =[self.titles_list['date'][num],
                        str(self.titles_list['titles'][num]),
                        str(self.titles_list['url_titles'][num]),
                        str(self.titles_list['main_text'][num])]
                EstimateFaster(num,self.titles_list['url_titles'], 
                'number of news from NEW YOURK TIMES that are saved [scripy:{}]'.format(self.words))
                file = file.writerow(llist)
        print("*** Done! ***")
        df_Nytimes = pd.DataFrame(self.titles_list)
        #df_CENT.head()
        return df_Nytimes
    
    
    

"-----------------------------------------------------------------------------"


class marketwatch_news_extract():
    
    def __init__(self,words):
        self.words = [x.lower() for x in words]
        self.titles_list = {}
        self.titles_list['date'] = []
        self.titles_list['titles'] = []
        self.titles_list['url_titles'] = []
        self.titles_list['main_text']= []
        
   
        
    def marketwatch_news_topics(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with issue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
    
        
    
        url1  = 'https://www.marketwatch.com'
        req = requests.get(url1)
        soup = BeautifulSoup(req.text,'lxml')
    
 
        titles1 = soup.find_all('h3')
        for title in titles1:
            tit = title.find('a',{'class':"link"} , href=True )
            if tit != None:
                if len(tit.text.split())>3:
                    for word in self.words:
                        m = re.search(word,tit.text.lower())
                        if m is not None:
                            self.titles_list['titles'].append(tit.text)
                            self.titles_list['url_titles'].append(tit['href'])
                            
            
            
            
        titles2 = soup.find_all('li',{'class':"list__item"})

        for title in titles2:
            tit = title.find('a',{'class':"link"} , href=True )
            if tit != None:
                if len(tit.text.split())>4:
                    for word in self.words:
                        m = re.search(word,tit.text.lower())
                        if m is not None:
                            self.titles_list['titles'].append(tit.text)
                            self.titles_list['url_titles'].append(tit['href'])
                            
        
    def marketwatch_news_search_subscript(self):
        """
        funcation:
        get url and go to page and get all of main text of page relate with issue
        ______________________________________
        input:
             titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
        ______________________________________
        output:
            titles_list: (dic, str) 
             1. url of titles news that match with words list
             2. title of news that match with words list
             3. main text relate to title and url
        """
        
        for num,url in enumerate(self.titles_list['url_titles']):
            EstimateFaster(num,self.titles_list['url_titles'], 'number of news on CNET that are found ')
        
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'lxml')
            body = soup.find_all('p')
            
            main_text = []
            for txt in body:
                txt = re.sub('\n','',txt.text)
                txt = re.sub('\r','',txt)
                txt = re.sub(' +',' ',txt)
                main_text.append(txt)
                
            main_text = ' '.join(main_text[1:len(main_text)-3])
            date = soup.find('time',{'class':"timestamp"})
            if date != None:
                
                date = re.sub('\n','',date.text)
                self.titles_list['main_text'].append(main_text)
                self.titles_list['date'].append(date)
            else:
                date = soup.find('p',{'id':"published-timestamp"}) 
                date = re.sub('Posted ','',date.text)
                self.titles_list['main_text'].append(main_text)
                self.titles_list['date'].append(date)
        
    def marketwatch_save(self,nameSave='data/marketwatch_new.csv'):
        """
        funcation:
            it search in https://www.marketwatch.com News with speacial keywords.
            it is made from two fucnation,
                1. Yahoo_finacial_search_topic
                2. Yahoo_finacial_search_subsript
        ______________________________________
        input:
            words: (list,str) it is a list of words we looking for them in titles of news
        ______________________________________
        output: 
             1. a csv file of news that it collected 
             2. give a dictounary from news (title, url, main_text)
        """
   
        #titles_list1 = delayed(Yahoo_finacial_search_topic)(self.words)
        #main_text = delayed(Yahoo_finacial_search_subscript)(titles_list1)
        #self.titles_list = main_text.compute()
        
      
        with open (nameSave,'a') as f:
            for num,col in enumerate(self.titles_list['titles']):
                
                file = csv.writer(f)
                
                llist =[self.titles_list['date'][num],
                        str(self.titles_list['titles'][num]),
                        str(self.titles_list['url_titles'][num]),
                        str(self.titles_list['main_text'][num])]
                EstimateFaster(num,self.titles_list['url_titles'], 
                'number of news that are saved from marketwatch[scripy:{}]'.format(self.words))
                file = file.writerow(llist)
        print("*** Done! ***")
        df_marketwatch = pd.DataFrame(self.titles_list)
        #df_CENT.head()
        return df_marketwatch