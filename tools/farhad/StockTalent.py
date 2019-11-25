#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:49:50 2019

@author: Farhad
"""
import pandas as pd
from farhad.time_estimate import EstimateFaster
import os
import pandas_datareader as web
import datetime

def normalize_df(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return df_norm

def Normlize_nMonth(df,numday=90,Date="Date",price="Adj Close"):
    #df['month'] = df[Date].apply(lambda x: datetime.datetime.strptime(str(x),"%Y-%m-%d").month)

    lenght = int(round(len(df[price])/numday))-1
    Plist = [[] for i in range(lenght)]
    new_df = pd.DataFrame()
    print(lenght)
    for num  in range(0,lenght):
        Plist[num] = [x for x in (df[price][numday*num:(numday*(num+1))])]
        new_df[str(num)] = Plist[num] 
        EstimateFaster(num,Plist[num],'Normalize')
        new_df['norm_{}'.format(num)] = normalize_df(new_df[str(num)])
        
    Plist2 = pd.Series([x for x in (df[price][numday*lenght:])])
    Plist2 = normalize_df(Plist2)
        
    frames = [new_df['norm_{}'.format(num)] for num in range(lenght)]
    x = pd.concat(frames,axis=0)
    frames = [x, Plist2]
    result = pd.concat(frames, names=['index','norm']).sample(frac=1).reset_index(drop=True)
    print('*** Done! ***')
    return  result

def Extract_month(x):
     return datetime.datetime.strptime(str(x),"%Y-%m-%d").month
 
def label_according_model_one(df,target="norm_Adj_price"):
    def Give_lebal_according_price(x):
        if x>=0.2:
            y=2
        elif x>=0:
            y=1
        elif x<0:
            y=-1
        elif x<=-0.2:
            y=-2
        return y
    df["Price_label"] = df[target].apply(Give_lebal_according_price)
    
    return df

def label_price_small(x):
    if x>=0:
        y=1
    elif x<0:
        y=0
    else:
        y=x
    return y

def label_according_model_two(df,target="norm_Adj_price"):
    def Give_lebal_according_price(x):
        
        if x>=0:
            y=1
        elif x<0:
            y=0
        
        return y
    df["Price_label_(0,1)"] = df[target].apply(Give_lebal_according_price)
    
    return df

def give_label_price_model_one(df_data, date= 'created_at', company='TSLA', address='data/stock_price_tesla_from2007.csv'):
    
    
    """
    Get tweets dataframe, then download price of stock marekt according to company_input
    after that, normlize Adj price and extarct label accroding one model (0,1), 
    finally, five a datafram of tweets with price_label(0,1)  
    ---------------------------------------------------
    inputs:
            df_data
            date= 'created_at'
            company='TSLA'
            address='data/stock_price_tesla_from2007.csv'
    ---------------------------------------------------
    outputs:
            a new Dataframe
    """
    
    start = df_data.sort_values(date)[date][0]
    end = df_data.sort_values(date)[date][len(df_data)-1]
    
    
    
    try:
        #yf.pdr_override()
        df_price =web.DataReader('TSLA','yahoo',start ,end)
    except:
        print("Can't take frish data from internet, used old")
        df_price = pd.read_csv(address)
    #print(len(df_price))
    
    
    result = Normlize_nMonth(df_price,90, Date="Date", price="Adj Close")
    df_price['norm_Adj_price'] = [x for x in result ]
    
    df_price["Price_label(0,1)"] = df_price["norm_Adj_price"].apply(label_price_small)
    ##df_price = df_price.reset_index()
    df_price['Dates'] =  pd.to_datetime(df_price.index).date
    df_price = df_price.reset_index()
    df_price2 = df_price[['Date','norm_Adj_price','Price_label(0,1)']]
    try:
        new_df = pd.merge(left=df_data, left_on='created_at', right=df_price, right_on='Dates')
    except:
        new_df = pd.merge(left=df_data, left_on='created_at', right=df_price, right_on=df_price['Dates'])
    new_df = new_df[['created_at','text','Price_label(0,1)']]
    
    print('lenght of label_tweets: ', len(new_df))
    print('*** Done Merge ***')
    return new_df 



def Read_all_csv_file_in_aFolder(address_folder, date='created_at'):
    """
    get file_path and extratc all csv file, then: 
    contact them, 
    sort valuse according to date column,
    reset index and finally  give out a new Dataframe 
    --------------------------------------------
    inputs:
           address_folder = file that we want to ectratc csv files 
    --------------------------------------------
    ouputs:
            One new dataframe from join all csv files
    --------------------------------------------
    Note:
        All csv file should have same format (column name)
    """
    Files = os.listdir(address_folder)
    for num  in range(len(Files)):
    
        if num==0:
            df0 = pd.read_csv(address_folder+Files[num])
            df1 = pd.read_csv(address_folder+Files[num+1])
            new_df = pd.concat([df0,df1],axis=0)
        if num>1:
            df3 = pd.read_csv(address_folder+Files[num])
            new_df = pd.concat([new_df,df3],axis=0)
            
    new_df = new_df.sort_values(date)
    new_df.drop_duplicates(inplace=True)
    new_df.reset_index(inplace=True,drop=True)
    time1 = new_df.sort_values(date)[date][0]
    time2 = new_df.sort_values(date)[date][len(new_df)-1]
    
    print('lenght of dataframe: ',len(new_df))
    print('Start from:',time1)
    print('Until:',time2)
    return new_df