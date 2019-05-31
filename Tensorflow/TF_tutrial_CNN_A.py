
import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split

from farhad.utility import encode_text_dummy_array
from farhad.time_estimate import EstimateFaster

import numpy as np 
path_tb = 'log/6/'
def conv2d(inputs, channel_in, channel_out,name):
    #
    with tf.name_scope('conv2D_{}'.format(name)):
        w = tf.Variable(tf.zeros(shape=[3,3,channel_in,channel_out]),name='w_conv')
        b = tf.Variable(tf.zeros(shape=[channel_out]), name='b_conv')
        conv2 = tf.nn.conv2d(inputs, w, strides=[1,1,1,1], padding="SAME", name='conv2d_conv')
        act = tf.nn.relu(conv2+b)
        
        tf.summary.histogram(path_tb+"w_conv"+name, w)
        tf.summary.histogram(path_tb+"b_conv"+name,b)
        tf.summary.histogram(path_tb+"act_conv"+name,act)
        
        return act
    
def feedforward(inputs, units , name):
    with tf.name_scope('feedforeard{}'.format(name)):
        w = tf.Variable(tf.zeros(shape=[inputs.shape[1], units]),name='w_ff')
        b = tf.Variable(tf.zeros(shape=[units]),name='b_ff')
        ma = tf.matmul(inputs,w)
        act = tf.nn.relu(ma+b, name="act_ff")
        
        tf.summary.histogram(path_tb+"w_ff"+name, w)
        tf.summary.histogram(path_tb+"b_ff"+name,b)
        tf.summary.histogram(path_tb+"act_ff"+name,act)
        
        return act
            
def write_graph(sess, name):
    join = os.path.join(path_tb, name)
    file_writer = tf.summary.FileWriter(join)
    file_writer.add_graph(sess.graph)
    
class Dataset:
    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass


    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


class mnist_model():
    def __init__(self,intputs):
        self.sess =  tf.Session()
        
       
        self.pred = ''
        
        self.cross_entropy= ''
        self.train_step =''
        self.accuracy = ''
        self.summ =""
        self.writer = ""
        self.intputs = intputs
    
        self.x = tf.placeholder(shape=[None,28,28],dtype=tf.float32, name='as_inputs')
        self.x_image = tf.reshape(self.x, [-1,28,28,1])
        tf.summary.image(path_tb+'as_inputs',self.x, 3)
    
        self.y = tf.placeholder(shape=[None,10],dtype=tf.float32, name="as_outputs")
    
        conv_1 = conv2d(inputs=self.x_image, channel_in=1, channel_out=32,name="conv1")
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    
        Flatten_shape = [-1, (14)*(14)*32] #14/2
        Flatten_1 = tf.reshape(pool_1, shape=Flatten_shape , name="Flatten")
    
        ff_1 = feedforward(Flatten_1, units=128, name="ff_1")
        ff_2 = feedforward(ff_1, units=10, name="ff_2")
        ff_3 = tf.nn.softmax(ff_2, name="softmax_layer")
    
        tf.summary.histogram(path_tb+'softmax',ff_3)
        self.pred = ff_3
        
    
    def compiles(self,learning_rate=1e-4):
        
        with tf.name_scope('Losses_funcation'):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.y), name="cross_entropy")
            tf.summary.scalar(path_tb+'cross_entropy', self.cross_entropy)
            
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
            
        
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   
            tf.summary.scalar(path_tb+"accuracy", self.accuracy)
        
        self.summ = tf.summary.merge_all() 
        self.saver = tf.train.Saver()        
        
        self.sess.run(tf.global_variables_initializer())
        write_graph(self.sess, 'model')
        
    def make_hparam_string(learning_rate, use_two_fc=False, use_two_conv=False):
        conv_param = "conv=2" if use_two_conv else "conv=1"
        fc_param = "fc=2" if use_two_fc else "fc=1"
        return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)
    
    def next_batch(self,num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    def fit(self, inputs, outputs, validation_size=0.20 , epoch=1 ,verbose=0 , batch=30):
        
        data = inputs,outputs
        X_train, X_test, y_train, y_test = train_test_split( inputs,outputs , test_size=validation_size)
        X_train_1,  y_train_1 = Dataset(X_train), Dataset(y_train)
        for i in range(epoch+1):
            X_train,  y_train  = X_train_1.next_batch(batch),  y_train_1.next_batch(batch)
            if i % 5 == 0:
                [train_accuracy, s] = self.sess.run([self.accuracy, self.summ], feed_dict={self.x: X_train, self.y: y_train})
                
            if  verbose==1 and i % 5 == 0:
                con = int((i+1)*50/epoch)
                con_ant = 50-con
                #run = "[step:"+str(i)+"/"+str(epoch)+"|"+"|"+"[training accuracy:"+str(train_accuracy)+"] \n"
                sys.stdout.write("[step: {}/{}]|{}{}|[training accuracy: {}] \n".format(i,epoch,con*'#',con_ant*' ',train_accuracy))
                #sys.stdout.write(run)
            if i % 10 == 0:
                #self.sess.run(feed_dict={self.x: X_test, self.y: y_test}) #assignment, 
                self.saver.save(self.sess, os.path.join(path_tb, "model.ckpt"), i)
                #write_graph(self.sess, str(i))
                
            self.sess.run(self.train_step, feed_dict={self.x: X_train, self.y: y_train})
            try:
                self.writer = tf.summary.FileWriter(path_tb + hparam)
            except:
                 self.writer = tf.summary.FileWriter(path_tb + 'model')
            EstimateFaster(i,epoch,"process is runing,training_accuracy:{}".format(train_accuracy))
        print('last accuracy:', train_accuracy)
        
