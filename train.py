import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import cv2
import skimage
import pickle as pkl
import h5py

import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter


from CaptionGenerator import Caption_Generator 

model_path = './models/tensorflow'

dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 256

n_epochs = 40

def preProBuildWordVocab(sentence_iterator, word_count_threshold=3): # function from Andre Karpathy's NeuralTalk
    print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent:
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
    f=open("data/voca.txt","w+")
    for w in vocab:
        f.write(("%s  %3d\n")%(w,word_counts[w]))
    f.close()
    ixtoword = {}
    ixtoword[0] = '$'  
    wordtoix = {}
    wordtoix['#START#'] = 0 
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1
    word_counts['$'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) 
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) 
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)

def new_get_data():
    h5 = h5py.File('data/image_vgg19_fc1_feature.h5', 'r')

    f = open('data/train.txt', encoding='utf-8')
    key = [key for key in h5.keys()]
    cnn = h5[key[1]].value

    f2 = open('data/valid.txt', encoding='utf-8')
    key = [key for key in h5.keys()]
    cnn2 = h5[key[2]].value

    i=-1
    new_cnn=[]
    annotations=[]
    for line in f:
        line=line.strip()
        if len(line)<=5:
            i=i+1
        else:
            # if len(line) <5:
            #     continue
            # if len(line) >25:
            #     continue
            line=line+"$"
            new_cnn.append(cnn[i])
            annotations.append(line)
    i=-1
    for line in f2:
        line=line.strip()
        if len(line)<=5:
            i=i+1
        else:
            # if len(line) <5:
            #     continue
            if len(line) >25:
                continue
            line=line+"$"
            new_cnn.append(cnn2[i])
            annotations.append(line)
    return np.array(new_cnn),np.array(annotations)

def train(learning_rate=0.001, continue_training=False):
    
    tf.reset_default_graph()

    # feats, captions = get_data(annotation_path, feature_path)
    feats, captions = new_get_data()
    # print(feats.shape)
    # print(captions.shape)
    # return 0
    wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)
    print(ixtoword)
    # return 0
    np.save('data/ixtoword', ixtoword)

    index = (np.arange(len(feats)).astype(int))
    np.random.shuffle(index)


    sess = tf.InteractiveSession()
    n_words = len(wordtoix)
    # maxlen = np.max( [x for x in map(lambda x: len(x.split(' ')), captions) ] )
    maxlen = np.max( [x for x in map(lambda x: len(x), captions) ] )
    print("n_words maxlen:",n_words,maxlen)
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, "train",init_b)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=100)
    global_step=tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                       int(len(index)/batch_size), 0.95)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.global_variables_initializer().run()

    if continue_training:
        saver.restore(sess,tf.train.latest_checkpoint(model_path))
    losses=[]
    for epoch in range(n_epochs):
        index = (np.arange(len(feats)).astype(int))
        np.random.shuffle(index)
        for start, end in zip( range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):

            current_feats = feats[index[start:end]]
            current_captions = captions[index[start:end]]
            current_caption_ind = [x for x in map(lambda cap: [wordtoix[word] for word in cap[:-1] if word in wordtoix], current_captions)]
            # print(current_captions[0])
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] )

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([x for x in map(lambda x: (x != 0).sum()+2, current_caption_matrix )])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats.astype(np.float32),
                sentence : current_caption_matrix.astype(np.int32),
                mask : current_mask_matrix.astype(np.float32)
                })

            print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs), "\t Iter {}/{}".format(start,len(feats)))
        print("Saving the model from epoch: ", epoch)
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)



if __name__ == '__main__':
    # pp = Preprocess()
    # cnn_train,cnn_valid,cnn_test,dic,caption_train,caption_valid = pp.preprocess()
    # print(caption_train)
    # feats, captions = get_data(annotation_path, feature_path)
    # print(feats.shape)
    # print(captions.shape)
    # print(captions[0])
    # feats, captions = get_data(annotation_path, feature_path)
    # print(feats.shape)
    # print(captions.shape)
    # wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)
    # feats, captions = get_data(annotation_path, feature_path)    
    # print(captions)
    # new_get_data()
    try:
        train(.001,False)
    except KeyboardInterrupt:
        print('Exiting Training')
