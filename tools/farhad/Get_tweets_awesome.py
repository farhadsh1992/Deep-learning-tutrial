#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 23:06:55 2019

@author: Farhad
"""

import tweepy
from farhad.farhadTwitterKey import farhadkey
import datetime
import pandas as pd
import sys

print('------------------------------------------------------------------------')
print("collect Tweets with spescial query")
print('Class: get_twitter()')
print('inpout: uery,item,namesave ')
print('  ')
print('def1: authenticate_twitter_app()')
print('def2: get_by_query_until()')
print('  ')
print('outpout: df of tweets, csv file')
print('------------------------------------------------------------------------')



class get_twitter():
    
    """
    use Twiiter API (by tweepy api)
    
    """
    
    
    def __init__(self,query,item,namesave):
        self.df_traing = pd.DataFrame()
        #self.start_date = start_date
        self.query = query
        self.namesave = namesave
        self.item = item
    def __help__():
        pass
    def __version__():
        print("1.1.12")
        
    
    
        
    def authenticate_twitter_app(self):
        key = farhadkey()
        consumer_key = key["CONSUMER_KEY"]
        consumer_secret = key["CONSUMER_SECRET"]
        access_token = key["ACCESS_TOKEN"]
        access_secret = key["ACCESS_SECRET"]
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(auth)
        
        
    def get_by_query_until(self):
        data_name = []
        data_screen_name =[]
        data_location = []
        data_text =[]
        data_created_at= []
        data_geo = []
        data_source =[]
        data_idtwitter =[]
      
       
        end_date = datetime.datetime.now()
        
       
        for number,status in  enumerate(tweepy.Cursor(self.api.search,
                                                      q=self.query,
                                                      result_type="recent",
                                                      lan='en' ).items(self.item)):
            
                
                data_name.append(status.user.name)
                data_screen_name.append(status.user.screen_name)
                data_location.append(status.user.location)
                data_text.append(status.text)
                data_created_at.append(status.user.created_at)
                data_geo.append(status.coordinates)
                data_source.append(status.source)
                #data_idtwitter.append(idtwitter)
                run = ("[Number of Tweets have been gotten:"+str(number+1)+"]["+str('collected tweets')+']')
                sys.stdout.write('\r'+ run)
                   
                    
        self.df_traing['name'] = data_name
        self.df_traing['screen_name'] = data_screen_name
        self.df_traing['text'] = data_text
        self.df_traing['created_at'] = data_created_at
        self.df_traing['geo'] = data_geo
        self.df_traing['source'] = data_source
        self.df_traing['data_location'] = data_location
        
        
        
        self.df_traing.to_csv(self.namesave,index=False)
        print("*** DONE!(I am cool) ***")
        return self.df_traing