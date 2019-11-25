#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:50:39 2018

@author: Farhad

textblob

"""

from textblob import TextBlob
import multiprocessing
import pandas as pd

class editor():
    def __init__(self):
        self.edited = []
    def Editor(self,li):
        lA = TextBlob(str(li)).correct()
        self.edited.append(str(lA))
        df1 = pd.Series(self.ListC)
        df1.to_csv('temperray_file.csv')
        return self.edited