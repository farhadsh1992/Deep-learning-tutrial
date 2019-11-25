
from farhad.Farhadcolor import tcolors, bcolors
from nltk import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pickle


from nltk import word_tokenize
from nltk.corpus import stopwords
from glove import Glove
from glove import Corpus
import numpy as np



# https://www.kaggle.com/eswarbabu88/toxic-comment-glove-logistic-regression

class Fglove_nltk():
    

    def __info__(self):
        print(""" 
        ------------------------------------------------------------------- \n
        Funcation:
        0. model_GLove = Text_Embadding(df.text) 
        
        \n"""+tcolors.GREEN+'Choose one of dicationary below:'+tcolors.ENDC+
        """
        1. model_GLove.Glove_100k_6B_50d()
        1. model_GLove.Glove_twitter_27B_25d()
        1. model_GLove.Glove_twitter_27B_50d()
        1. Glove_datastories_twitter_50d(
        \n"""+tcolors.GREEN+'Next:'+tcolors.ENDC+"""
        2. data_embedding = model_GLove.data2vec()  
        3. sent_embedding = model_GLove.sent2vec(text) just for a sentences
        \n"""+tcolors.GREEN+'For saving file:'+tcolors.ENDC+""" \n
        4. model_GLove.save_file(name_file_save)
        ------------------------------------------------------------------------
        """
        )

    def __init__(self,file=''):
        self.embeddings_index = {}
        self.df_text = file
        self.data_glove = []
        self.path_file = "You haven't chosen, yet" 
        self.size = 50
        self.problem = {}

    def open_glove_file(self):
        if self.path_file == "You haven't chosen, yet":
            print(self.path_file)
        else:
            with open(self.path_file, encoding='UTF8') as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    try:
                        coefs = np.asarray(values[1:], dtype='float32')
                        self.embeddings_index[word] = coefs
                    except ValueError:
                        print(bcolors.RED,'[There is a problem]',bcolors.ENDC)
            f.close()
            print('Found %s word vectors.' % len(self.embeddings_index))

    def Glove_100k_6B_50d(self):
        self.path_file = '/anaconda3/lib/python3.6/farhad/data/Glove/glove.first-100k.6B.50d.txt'
        self.size = 50
    def Glove_twitter_27B_25d(self):
        self.path_file = '/anaconda3/lib/python3.6/farhad/data/Glove/glove.twitter.27B.25d.txt'
        self.size = 25
    def Glove_twitter_27B_50d(self):
        self.path_file = '/anaconda3/lib/python3.6/farhad/data/Glove/glove.twitter.27B.50d.txt'
        self.size = 50
    def Glove_datastories_twitter_50d(self):
        self.path_file = '/anaconda3/lib/python3.6/farhad/data/Glove/datastories.twitter.50d.txt'
        self.size = 50
    
    def sent2vec(self,x, number=0):
        words = str(x).lower()
        words = word_tokenize(words)
        #words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(self.embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            self.problem[number] = M
            return np.zeros(self.size)
        return v / np.sqrt((v ** 2).sum())

    def extract_vec(self):
        self.open_glove_file()
        self.data_glove = [self.sent2vec(x, num) for num,x in enumerate(self.df_text)]
        print('Lenght of data:',len(self.data_glove))
        print('lenght of features:',len(self.data_glove[11]))
        return self.data_glove

    def save_file(self,name_file_save):
        with open(name_file_save+'.pkl','wb') as f:
            pickle.dump( self.data_glove ,f,pickle.HIGHEST_PROTOCOL)

def Load_glove_nltk(name_file):
    with open(name_file,'rb') as f:
        return pickle.load(f)








class Fglove_glove():
    def __init__(self, cropus_path):
        self.problem = {}
        self.cropus_path = cropus_path
        self.size = 50
        self.model = ""
    def __info__(self):
        pass
    
    def Choose_knid_of_StanfordModel(self,resourse=('wikipedia'  or 'twitter'), dim=('100d' or '50d' or '25d')):
        if resourse=='wikipedia':
            if dim=='100d':
                pass
            elif dim=='50d':
                self.model_kind = '/anaconda3/lib/python3.6/farhad/data/Glove/glove.first-100k.6B.50d.txt'
                self.size = 50
            elif dim=='25d':
                pass
            
        elif resourse=='twitter':
            if dim=='100d':
                pass
            elif dim=='50d':
                self.model_kind = '/anaconda3/lib/python3.6/farhad/data/Glove/glove.twitter.27B.50d.txt'
                self.size = 50
            elif dim=='25d':
                self.model_kind = '/anaconda3/lib/python3.6/farhad/data/Glove/glove.twitter.27B.25d.txt'
                self.size = 25
               
        else:
            print("not existed as resourse in database \n you should choose other resourse or add your new dataset")
    
        
    def read_corpus(self,filename):
        """
           Read corpus from regular text file 
        """
    
        delchars = [chr(c) for c in range(256)]
        delchars = [x for x in delchars if not x.isalnum() ] 
        delchars.remove(' ')
        delchars = ''.join(delchars)
        table = str.maketrans(dict.fromkeys(delchars))
        try:
            with open(filename, 'r') as datafile:
                for line in datafile:
                    yield line.lower().translate(table).split(' ')
        except:
            for line in filename:
                yield line.lower().translate(table).split(' ')
            
    def Myself_Model(self, cropus_path, save=None, back_corpus=None,  epochs =10, no_threads = 8, no_components=100, learning_rate=0.05 ):
        """
        sd
        """
        
        self.get_data = self.read_corpus(cropus_path)
        corpus_model = Corpus()
        corpus_model.fit(self.get_data, window=10)
      

        #self.glove = Glove()
        self.glove = Glove(no_components=no_components, learning_rate=learning_rate)
        self.glove.fit(corpus_model.matrix, epochs=epochs, no_threads=no_threads, verbose=False)
        self.glove.add_dictionary(corpus_model.dictionary)
        
        if save!= None :
            #save = 'model/articles_glove.model'
            self.glove.save(save)
            
        self.model = self.glove

        if back_corpus != None:
            return corpus_model,self.glove 
        else:
            return  self.glove
    
    def glove_sent2vec(self,sent, model, number=0):
        words = str(sent).lower()
        words = word_tokenize(words)
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                ww = model.word_vectors[model.dictionary[w]][:]
                M.append(ww)
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            self.problem[number] = M
            return np.zeros(self.size)
        return v / np.sqrt((v ** 2).sum())
    
    def extract_vec(self, model):
        #self.open_glove_file()
        self.get_data = self.read_corpus(self.cropus_path)
        self.data_glove = [self.glove_sent2vec(sent = x, model = model, number=num) for num,x in enumerate(self.get_data)]
        print('Lenght of data:',len(self.data_glove))
        print('lenght of features:',len(self.data_glove[1]))
        return self.data_glove

    
    def Pre_Existing_Model(self,cropus_path):
        
        
        self.get_data = self.read_corpus(cropus_path)
        
        glove = Glove()
        stanford = glove.load_stanford(self.model_kind)
        self.model = stanford
        
        return stanford
    def most_similar(self,positive, negtive,dictionary,word_vectors , topn=10, freq_thersold=5, ):
        """
        build a mean vector model for the give positive and negtive terms,
        """
        vocab = dictionary
        w = word_vectors
    
        mean_vecs = []
        id2word = [i for i in vocab.keys()]
    
    
        for word in positive: mean_vecs.append(w[vocab[word]])
        for word in negtive: mean_vecs.append(-1* w[vocab[word]])
    
        mean = np.array(mean_vecs).mean(axis=0)
        mean /= np.linalg.norm(mean)
    
        # Now calculate cosine distance between this mean vector and all others,
        dists = np.dot(w, mean)
    
        best = np.argsort(dists)[::-1][:topn + len(positive) +len(negtive)+100]
    
    
        result = [(id2word[i], dists[i]) for i in  best if (vocab[id2word[i]] >= freq_thersold and id2word[i] not in positive and id2word[i] not in negtive) ]
     
        return result[:topn]    