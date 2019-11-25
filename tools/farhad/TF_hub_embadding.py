#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:48:26 2019

@author: Farhad
"""

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns

def First_Universal_Sentence_Encoder_embading(df):
    """
    it is fist function that I write for this, secound editition is faster
    The models take as input English strings and produce as output a fixed 512 dimensional embedding representation of the string. 
    -------------------------------------------------
    input: 
         dataframe of text
    -------------------------------------------------
    output: 
        embadding lsit of string in 512 dimensional
    -------------------------------------------------
    https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed
    !curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC sentence_wise_email/module/module_useT
    """
    
    embed = hub.Module("/Users/apple/Documents/Programming/python/Project/sentence_wise_email/module/module_useT")
    embeddings = embed(df)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        new_data = session.run(embeddings)
    return new_data

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
def Universal_Sentence_Encoder_embadding(df):
    """
    The encoder takes as input a lowercased PTB tokenized string and outputs a 512 dimensional vector as the sentence embedding. 
    -------------------------------------------------
    input: 
        dataframe of text
    -------------------------------------------------
    output: 
        embadding lsit of string in 512 dimensional
    -------------------------------------------------
    https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed
    !curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC sentence_wise_email/module/module_useT
    """
    
    embed_fn = embed_useT('/Users/apple/Documents/Programming/python/Project/sentence_wise_email/module/module_useT')
    data = embed_fn(df)
    return data
def nnlm_en_dim128_with_normalization(df):
    
    """
    The encoder takes as input a lowercased PTB tokenized string and outputs a 128 dimensional vector as the sentence embedding. 
    -------------------------------------------------
    input: 
        dataframe of text
    -------------------------------------------------
    output: 
        embadding lsit of string in 128 dimensional
    -------------------------------------------------
    https://tfhub.dev/google/nnlm-id-dim128-with-normalization/1?tf-hub-format=compressed
    !curl -L "https://tfhub.dev/google/nnlm-id-dim128-with-normalization/1?tf-hub-format=compressed" | tar -zxvC sentence_wise_email/module/module_useT2
    """
    
    
    embed_fn = embed_useT('/Users/apple/Documents/Programming/python/Project/sentence_wise_email/module/module_useT2')
    data = embed_fn(df)
    return data
def nnlm_en_dim50_with_normalization(df):
    """
    The encoder takes as input a lowercased PTB tokenized string and outputs a 50 dimensional vector as the sentence embedding. 
    -------------------------------------------------
    input: 
         dataframe of text
    -------------------------------------------------
    output: 
        embadding lsit of string in 50 dimensional
    -------------------------------------------------
    https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1
    """
    
    embed_fn = embed_useT('https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1')
    data = embed_fn(df)
    return data