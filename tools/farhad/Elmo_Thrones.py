"""
ISSUE: ELMO EMBEDDING
BY: Farhad 
CREATED_AT: 6 Apr 2019
(version 2 of TF_hub_embadding )

"""


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
from farhad.Farhadcolor import tcolors,bcolors



class ELMO_AWS():
    """
    https://tfhub.dev/s?module-type=text-embedding
    """
    def __init__(self):
        pass
    def __englishurl__(self):
        self.universal_sentence_encoder512 = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.universal_sentence_encoder_large512 = "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed"
        
        self.nnlm_en_dim128 = "https://tfhub.dev/google/nnlm-en-dim128/1"
        self.nnlm_en_dim128_with_normalization = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
        self.random_nnlm_en_dim128 = "https://tfhub.dev/google/random-nnlm-en-dim128/1"
        
        
        self.nnlm_en_dim50_with_normalization = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1"
        self.nnlm_en_dim50 = "https://tfhub.dev/google/nnlm-en-dim50/1"
        
    def __info__(self,mode=('dim512' or 'dim128' or 'dim50')):
        info = {}
        info['dim512']="""
        The encoder takes as input a lowercased PTB tokenized string and outputs
        a 512 dimensional vector as the sentence embedding. 
        -------------------------------------------------
        input: 
            dataframe of text
        -------------------------------------------------
        output: 
            embadding lsit of string in 512 dimensional
        -------------------------------------------------
        https://tfhub.dev/google/universal-sentence-encoder/2
        https://tfhub.dev/google/universal-sentence-encoder-large/3
        https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed
        !curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | 
                                                              tar -zxvC sentence_wise_email/module/module_useT
        """
        
        info['dim128']="""
        The encoder takes as input a lowercased PTB tokenized string and outputs 
        a 128 dimensional vector as the sentence embedding. 
        -------------------------------------------------
        input: 
            dataframe of text
        -------------------------------------------------
        output: 
            embadding lsit of string in 128 dimensional
        -------------------------------------------------
        https://tfhub.dev/google/nnlm-en-dim128/1
        https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1
        https://tfhub.dev/google/random-nnlm-en-dim128/1
        !curl -L "https://tfhub.dev/google/nnlm-id-dim128-with-normalization/1?tf-hub-format=compressed" | 
                                                                   tar -zxvC sentence_wise_email/module/module_useT2
        """
        
        info['dim50']="""
        The encoder takes as input a lowercased PTB tokenized string and outputs 
        a 50 dimensional vector as the sentence embedding. 
        -------------------------------------------------
        input: 
             dataframe of text
        -------------------------------------------------
        output: 
            embadding lsit of string in 50 dimensional
        -------------------------------------------------
        https://tfhub.dev/google/nnlm-de-dim50-with-normalization/1
        https://tfhub.dev/google/nnlm-en-dim50/1
        """
        
        print(info[mode])
        
    def __Embeddding__(self,module):
        
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            self.embed = hub.Module(module)
            self.embeddings = self.embed(sentences)
            self.session = tf.train.MonitoredSession()
            return lambda x: self.session.run(self.embeddings, {sentences: x})
    def __url__(self):
        self.__englishurl__()
        self.url_dictionary={}
        self.url_dictionary['dim512'],self.url_dictionary['dim128'],self.url_dictionary['dim50'] = {},{},{}
        
        
        self.url_dictionary['dim512']['norm'] = [self.universal_sentence_encoder_large512,
                                        "a 512 dimensional vector as the sentence embedding. "]
        self.url_dictionary['dim128']['norm'] = [self.nnlm_en_dim128_with_normalization,
                                         "a 128 dimensional vector as the sentence embedding."]
        self.url_dictionary['dim50']['norm']  = [self.nnlm_en_dim50_with_normalization,
                                        "a 50 dimensional vector as the sentence embedding."]
        
        self.url_dictionary['dim512']['Neural'] = [self.universal_sentence_encoder512,
                                        "a 512 dimensional vector as the sentence embedding. "]
        self.url_dictionary['dim128']['Neural'] = [self.nnlm_en_dim128,
                                         "a 128 dimensional vector as the sentence embedding."]
        self.url_dictionary['dim50']['Neural']  = [self.nnlm_en_dim50,
                                        "a 50 dimensional vector as the sentence embedding."]
        
        
    def Universal_Encoder(self,df,mode=('dim512' or 'dim128' or 'dim50'),kind="norm"):
        #if kind!='norm' or kind!='Neural':
            #return print(tcolors.RED,"You should choose norm or Neural for kind",tcolors.ENDC)
            
        self.__url__()
        embed_fn = self.__Embeddding__(self.url_dictionary[str(mode)][str(kind)][0])
        print(tcolors.GREEN+self.url_dictionary[str(mode)][str(kind)][1]+tcolors.ENDC)
        self.data = embed_fn(df)
        return self.data
    