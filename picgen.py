import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import cv2
import skimage

import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter

model_path = './models/tensorflow'
vgg_path = './data/vgg16-20160129.tfmodel'

image_path = './image_path.jpg'
from CaptionGenerator import Caption_Generator 

model_path = './models/tensorflow'

dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
learning_rate = 0.001
model_num=39


def crop_image(x, target_height=227, target_width=227, as_float=True):
    image = cv2.imread(x)
    if as_float:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))




def read_image(path):
    # print(path)
    img = crop_image(path, target_height=224, target_width=224)
    if img.shape[2] == 4:
        img = img[:,:,:3]

    img = img[None, ...]
    return img

def test_new(sess,image,generated_words,ixtoword):
    files=os.listdir('./test/')
    print(files)
    for file in files:
        # print('./test/'+file)
        test(sess,image,generated_words,ixtoword,'./test/'+file)

def test(sess,image,generated_words,ixtoword,test_image_path=0): # Naive greedy search

    

    feat = read_image(test_image_path)
    fc7 = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={images:feat})
    saver = tf.train.Saver()
    sanity_check=False
    # sanity_check=True
    if not sanity_check:
        saved_path=tf.train.latest_checkpoint(model_path)
        saver.restore(sess, './models/tensorflow/model-'+ str(model_num))
    else:
        tf.global_variables_initializer().run()

    generated_word_index= sess.run(generated_words, feed_dict={image:fc7})
    generated_word_index = np.hstack(generated_word_index)
    generated_words = [ixtoword[x] for x in generated_word_index]
    # print(generated_words)
    punctuation = np.argmax(np.array(generated_words) == '$')+1

    generated_words = generated_words[:punctuation]
    generated_sentence = ''.join(generated_words)
    print(test_image_path,generated_sentence)

if __name__ == '__main__':
    if not os.path.exists('data/ixtoword.npy'):
        print ('You must run 1. O\'reilly Training.ipynb first.')
    else:
        tf.reset_default_graph()
        with open(vgg_path,'rb') as f:
            fileContent = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fileContent)
        images = tf.placeholder("float32", [1, 224, 224, 3])
        tf.import_graph_def(graph_def, input_map={"images":images})
        ixtoword = np.load('data/ixtoword.npy').tolist()
        n_words = len(ixtoword)
        maxlen=25
        graph = tf.get_default_graph()
        sess = tf.InteractiveSession(graph=graph)
        caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)
        graph = tf.get_default_graph()
        image, generated_words = caption_generator.build_generator(maxlen=maxlen)
        test_new(sess,image,generated_words,ixtoword)

