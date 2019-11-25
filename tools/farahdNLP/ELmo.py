import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class ELMo_google_model():

    def __init__(self, df, name='text',dimension=50):
        
        self.sentences = np.array([x for x in df[name]])
        self.df_data = df
        self.x = []
        self.dimension = dimension
        #self.ELMo_Embadding()
        #self.reducation_dimensional()
        
    def ELMo_Embadding(self):
        url = "https://tfhub.dev/google/elmo/2"
        embed = hub.Module(url)

        embeddings = embed(
            self.sentences,
            signature="default",
            as_dict=True)["default"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            self.x = sess.run(embeddings)

        pca = PCA(n_components=self.dimension)
        self.x = pca.fit_transform(self.x)
        return self.x
    def reducation_dimensional(self,):
        pca = PCA(n_components=self.dimension) #reduce down to 50 dim
        self.x = pca.fit_transform(self.x)


    def Save_ELMo(self,namefile):
        self.df_data['Embadding_Elmo'] = [i for i  in self.x] 
        self.df_data.to_csv(namefile)

