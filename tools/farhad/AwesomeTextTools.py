#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:56:21 2019
issue Awesome Text Tools
@author: Farhad
"""

#from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import pandas as pd
import numpy as np
import glove
import os
from farhad.Farhadcolor import tcolors, bcolors

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def _info_():
    print('---------------------------------------------------------')
    print('*** First def ***')
    print('text_into_sequence_of_integers')
    print('')
    print('inputs: croups_df, dictunary, model="tfidf"')
    print('mode : ("binary" or "count" or "tfidf" or "freq")')
    print('')
    print('outputs: ')
    print('         Embadding = text to integer')
    print('         Sequence; test to mode, mode=("binary" or "count" or "tfidf" or "freq")')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('*** second def ***')
    print('text_into_sequence_of_integers_with_glove_dictionary')
    print('')
    print('inputs: croups_df,  model="tfidf"')
    print('mode : ("binary" or "count" or "tfidf" or "freq")')
    print('')
    print('outputs: ')
    print('         Embadding = text to integer')
    print('         Sequence; test to mode, mode=("binary" or "count" or "tfidf" or "freq")')
    print('---------------------------------------------------------')

# this is ok now : 
def text_into_sequence_of_matrix(sample_df, dictunary, model="tfidf"):
    """
    Funcation:
        allows to vectorize a text corpus, by turning each text into either a sequence of integers 
        (each integer being the index of a token in a dictionary) or into a vector where 
        the coefficient for each token could be binary, based on word count,
        more information: https://docs.w3cub.com/tensorflow~python/tf/keras/preprocessing/text/tokenizer/
        used : 'tensorflow.keras.preprocessing.text.Tokenizer'   Api 
    ---------------------------------------------------------------------------------
    inputs:
        sample_df; text list for transform
        dictunary; dictunary for  refrance words
        model = for mode of sequence
    ---------------------------------------------------------------------------------
    outputs:
        Embadding = text to integer
        sequence; test to mode, mode=("binary" or "count" or "tfidf" or "freq")
    
    """
    
    
    
    token = Tokenizer(num_words=1000,lower=True,split=" ",filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') 
    
   
    token.fit_on_texts(dictunary)
    sequence = token.texts_to_matrix(sample_df, mode = model)
    
    Embadding = token.texts_to_sequences(sample_df)
    one_hot_result = token.sequences_to_matrix(Embadding , mode = model)
    
    word_index = token.word_index
    print(tcolors.GREEN,'found {} unique tokens.'.format(len(word_index)),tcolors.ENDC)

    return one_hot_result, sequence

def text_into_sequence_of_integers(sample_df, dictunary, model="tfidf"):
    """
    Funcation:
        allows to vectorize a text corpus, by turning each text into either a sequence of integers 
        (each integer being the index of a token in a dictionary) or into a vector where 
        the coefficient for each token could be binary, based on word count,
        more information: https://docs.w3cub.com/tensorflow~python/tf/keras/preprocessing/text/tokenizer/
        used : 'tensorflow.keras.preprocessing.text.Tokenizer'   Api 
    ---------------------------------------------------------------------------------
    inputs:
        sample_df; text list for transform
        dictunary; dictunary for  refrance words
        model = for mode of sequence
    ---------------------------------------------------------------------------------
    outputs:
        Embadding = text to integer
        sequence; test to mode, mode=("binary" or "count" or "tfidf" or "freq")
    
    """
    
    
    
    token = Tokenizer(num_words=1000,lower=True,split=" ",filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') 
    
   
    token.fit_on_texts(dictunary)
    #sequence = token.texts_to_matrix(croups_df, mode = model)
    
    sequence = token.texts_to_sequences(sample_df)
    #sequence = token.sequences_to_matrix(Embadding , mode = model)
    one_hot_result = token.texts_to_matrix(sample_df)
    word_index = token.word_index
    print(tcolors.GREEN,'found {} unique tokens.'.format(len(word_index)),tcolors.ENDC)

    return one_hot_result, sequence

def text_into_sequence_of_integers_with_glove_dictionary(croups_df, model="tfidf"):
    """
    Funcation:
        text_into_sequence_of_integers_with_glove_dictionary
        allows to vectorize a text corpus, by turning each text into either a sequence of integers 
        (each integer being the index of a token in a dictionary) or into a vector where 
        the coefficient for each token could be binary, based on word count,
        more information: https://docs.w3cub.com/tensorflow~python/tf/keras/preprocessing/text/tokenizer/
        used : 'tensorflow.keras.preprocessing.text.Tokenizer'   Api 
    ---------------------------------------------------------------------------------
    inputs:
        croups_df; text list for transform
        dictunary; dictunary for  refrance words
        model = for mode of sequence
    ---------------------------------------------------------------------------------
    outputs:
        Embadding = text to integer
        sequence; test to mode, mode=("binary" or "count" or "tfidf" or "freq")
    
    """
    
    token = Tokenizer(num_words=1000,lower=True,split=" ",filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') 
    glove_dir = '/anaconda3/lib/python3.6/farhad/data/RNN/'
    glove_100k_50d = 'glove.first-100k.6B.50d.txt'
    glove_100k_50d_path = os.path.join(glove_dir, glove_100k_50d)
    
    word_embedding = glove.Glove.load_stanford( glove_100k_50d_path ).dictionary
    token.fit_on_sequences(word_embedding)
    token.fit_on_texts(word_embedding)
    try:
        sequence = token.sequences_to_matrix(croups_df , mode = model)
    except:
        sequence = token.texts_to_matrix(croups_df, mode = model)
    
    #Embadding = token.texts_to_sequences(croups_df)
    #sequence = token.sequences_to_matrix(Embadding , mode = model)
    
    return word_embedding, sequence



def Get_embeddings_from_Glove(df,max_words,vocabulary_size = 20000,glove_path=None):
    embeddings_index = dict()
    
    
    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(df)
    sequences = tokenizer.texts_to_sequences(df)
    data = pad_sequences(sequences, maxlen=max_words)
    if glove_path == None:
        glove_dir = '/anaconda3/lib/python3.6/farhad/data/RNN/'
        glove_100k_50d = 'glove.first-100k.6B.50d.txt'
        glove_path = os.path.join(glove_dir, glove_100k_50d)
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # Create a weight matrix for words in training docs
    seq = [x.split() for x in df]
    dic = []
    for x in seq:
        for i in x:
            dic.append(i)
    dic = set(dic)
    embedding_matrix = np.zeros((vocabulary_size, max_words))
    for word, index in tokenizer.word_index.items():
        
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
                
    return embedding_matrix



def Visualize_Embedding_in_TensorBoard(word_embedding):
    
    embedding_var = tf.Variable(word_embedding , dtype='float32', name='word_embedding')
    projector_config = projector.ProjectorConfig()
    
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    
    LOG_DIR='tensorflow.logdir/'
    os.makedirs(LOG_DIR, exist_ok=True)
    
    metadata_file = 'glove_full_50d.words.tsv'
    vocab_list = [ word_embedding.inverse_dictionary[i] for i in range(len( word_embedding.inverse_dictionary )) ]
    
    with open(os.path.join(LOG_DIR, metadata_file), 'wt') as metadata:
        metadata.writelines("%s\n" % w for w in vocab_list)
        
    embedding.metadata_path = os.path.join(os.getcwd(), LOG_DIR, metadata_file)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    projector.visualize_embeddings(summary_writer, projector_config)
    
    saver = tf.train.Saver([embedding_var])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(LOG_DIR, metadata_file+'.ckpt'))
        
    print("Look at the embedding in TensorBoard : http://localhost:8081/")