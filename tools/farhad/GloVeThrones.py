




from nltk import word_tokenize
from nltk.corpus import stopwords
from glove import Glove
from glove import Corpus
import numpy as np


class Fglove_glove():
    def __init__(self):
        self.problem = {}
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
        if back_corpus != None:
            yield corpus_model

        #self.glove = Glove()
        self.glove = Glove(no_components=no_components, learning_rate=learning_rate)
        self.glove.fit(corpus_model.matrix, epochs=epochs, no_threads=no_threads, verbose=True)
        self.glove.add_dictionary(corpus_model.dictionary)
        
        if save!= None :
            #save = 'model/articles_glove.model'
            self.glove.save(save)
            
        self.model = self.glove
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
        self.data_glove = [self.glove_sent2vec(x, num, model) for num,x in enumerate(self.get_data)]
        print('Lenght of data:',len(self.data_glove))
        print('lenght of features:',len(self.data_glove[1]))
        return self.data_glove

    
    def Pre_Existing_Model(self,cropus_path):
        
        
        self.get_data = self.read_corpus(cropus_path)
        
        glove = Glove()
        stanford = glove.load_stanford(self.model_kind)
        self.model = stanford
        
        return stanford