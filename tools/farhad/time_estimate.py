#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 18:06:32 2018

@author: Farhad
"""
import datetime
import pendulum
import psutil
import sys
import os



def TimeEstimate(start,num,xlist):
    num+=1

   
    stop = pendulum.now()
    dift1 = abs(start-stop)*1
    estimatTime = abs(abs(dift1*len(xlist)/num)-dift1)
                 
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = psutil.virtual_memory().percent
        
    cpu = psutil.cpu_percent(interval=1, percpu=True)
    run = ("["+str(num)+'/'+str(len(xlist))+"][time:" + str(dift1) +"]"+
             "[Remain: " + str(estimatTime)+
             "]["+" Memory: "+ 
             str(memoryUse)+"%]["+"CPU:"+
             str(cpu)+"%]")
        
    sys.stdout.write('\r'+ run)
    
    
def EstimateFaster(num,len_t, description):
    num+=1
    run = ("["+str(num)+'/'+str(len_t)+"]["+str(description)+']')
        
    sys.stdout.write('\r'+ run)
