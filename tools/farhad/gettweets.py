#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:35:20 2018

@author: Farhad
Title : get data from twitter
"""
import tweepy
from farhad.farhadTwitterKey import farhadkey

import datetime
import pendulum
import re
from farhad.time_estimate import TimeEstimate

import os
import psutil
import sys
import pandas as pd
import time

    
class __help__():
    
    def __init__():
        df = pd.DataFrame()
        df['funcation'] = ['authenticate_twitter_app','time_estimate','get_by_Luser_Lquery_until']
        df['input'] =['__non__','number int, start_time int', 
         'list_user list,query list,year=2018 int,month int,day int'] 
        df['output'] = ['api','time bar','dataframe from twitter']
        df.head(len(df))
class f_twitter():
    
    print("""
    pakges that used:  tweepy, farhad.farhadTwitterKey  ,keras.preprocessing.text    
    datetime ,pendulum, re, os,psutil, sys, pandas,time
    """)
    
    
    def __init__(self,df):
        self.df_traing = df
        
    def __help__():
        df = pd.DataFrame()
        df['funcation'] = ['authenticate_twitter_app','time_estimate','get_by_Luser_Lquery_until']
        df['input'] =['__non__','number int, start_time int', 
          'list_user list,query list,year=2018 int,month int,day int'] 
        df['output'] = ['api','time bar','dataframe from twitter']
        df.head(len(df))
    def __version__():
        print("1.1.12")
        
    
    def time_estimate(num,xlist, description):
        num+=1
        run = ("["+str(num)+'/'+str(len(xlist))+"]["+str(description)+']')
        sys.stdout.write('\r'+ run)
        
    def authenticate_twitter_app(self):
        key = farhadkey()
        consumer_key = key["CONSUMER_KEY"]
        consumer_secret = key["CONSUMER_SECRET"]
        access_token = key["ACCESS_TOKEN"]
        access_secret = key["ACCESS_SECRET"]
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(auth)
        return self.api
        
    def get_by_Luser_Lquery_until(self,listUser,query,year,month,day,namesave):
        data_name = []
        data_screen_name =[]
        data_location = []
        data_text =[]
        data_created_at= []
        data_geo = []
        data_source =[]
        data_idtwitter =[]
      
        old_length = 0
        firstnumber= 0
        untiltime = datetime.datetime(year,month,day)
        start = pendulum.now()
        
        for num,user in enumerate(listUser):
           
            if num == firstnumber:
                time.sleep(10)
                import tweepy
                
                api = self.api
                firstnumber +=5
            
            TimeEstimate(start,num,listUser)
            
            for number,status in  enumerate(tweepy.Cursor(api.user_timeline,
                                                          screen_name = user,
                                                          q=query,
                                                          result_type="recent",
                                                          until=untiltime,
                                                          include_entities=True,
                                                          lan='en' ).items()):
                souped = status.text
                lower_case = souped.lower()
                
                for num,i in enumerate(query):
                     if  i in  lower_case  :
                         data_name.append(status.user.name)
                         data_screen_name.append(status.user.screen_name)
                         data_location.append(status.user.location)
                         data_text.append(status.text)
                         data_created_at.append(status.user.created_at)
                         data_geo.append(status.coordinates)
                         data_source.append(status.source)
                         #data_idtwitter.append(idtwitter)
                time_estimate(num,query, 'looking for query')
                   
                    
        self.df_traing['name'] = data_name
        self.df_traing['screen_name'] = data_screen_name
        self.df_traing['text'] = data_text
        self.df_traing['created_at'] = data_created_at
        self.df_traing['geo'] = data_geo
        self.df_traing['source'] = data_source
        self.df_traing['data_location'] = data_location
        
        
        
        self.df_traing.to_csv("temperray_file_gettweets{}.csv".format(namesave))
        return self.df_traing
