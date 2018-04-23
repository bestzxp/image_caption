

import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import h5py
import codecs

import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter

from CaptionGenerator import Caption_Generator 

model_path = './models/tensorflow'


dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
model_num=22


def get_data():
    h5 = h5py.File('data/image_vgg19_fc1_feature.h5', 'r')
    key = [key for key in h5.keys()]

    cnn = h5[key[0]].value
    print(key[0],cnn.shape)
    return cnn
 
def test(sess,image,generated_word,ixtoword): 

    
    feats=get_data()
    generated_words=generated_word
    
    saver = tf.train.Saver()

    saver.restore(sess, './models/tensorflow/model-'+ str(model_num))
    f = codecs.open('result.txt', 'w+','utf-8-sig')
    for i in range(1000):
        feat = np.array([feats[i]])
        generated_word_index= sess.run(generated_words, feed_dict={image:feat})
        generated_word_index = np.hstack(generated_word_index)

        generated_sentence = [ixtoword[x] for x in generated_word_index]
        f.write(str(9000+i)+" ")
        flag=True
        for s in generated_sentence:
            if s=='$':
                f.write("\n")
                flag=False
                break
            f.write(s+" ")
        if flag:
            f.write("\n")
        if i%20==0:
            print(i,generated_sentence)
    

if __name__ == '__main__':

    
    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    maxlen=25
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)
    image, generated_words = caption_generator.build_generator(maxlen=maxlen)
    test(sess,image,generated_words,ixtoword)
