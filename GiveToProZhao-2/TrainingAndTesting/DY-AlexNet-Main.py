#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:05:08 2018

@author: scw4750
"""


################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import os
#from pylab import *
import numpy as np
from matplotlib import pyplot as plt
#from scipy.ndimage import filters
#import Queue
#from PIL import Image  #
import tensorflow as tf
import Function
#from caffe_classes import class_names

#%%
#TRAININGCLASSNUM = 0
#TRAININGSAMPLENUM = 0    #AmazonEasy=4453 AmazonAll= 12433

Epoch = 2
TrainingTotalBatchSize = 300
TrainingPart1BatchSize = 200
TrainingPart2BatchSize = 100
QuestBatchSize = 1


class_name = ['collar_design_labels',
              'neckline_design_labels',
              'skirt_length_labels',
              'sleeve_length_labels',
              'neck_design_labels', 
              'coat_length_labels',
              'lapel_design_labels',
              'pant_length_labels']

NOWCLASS = class_name[7]





AttrKeyCsvFilePath = 'Data/AttrKey//' + NOWCLASS + '_class' + '.csv'
TrainingInfoCsvFilePath = 'Data/CSVfiles/' + 'Train_Info_for_' + NOWCLASS + '.csv'
QuestInfoCsvFilePath = 'Data/CSVfiles/' + 'Quest_Info_for_' + NOWCLASS + '.csv'
QuestCsvFilePath = 'Data/' + 'question' + '.csv'


TrainingTotalTFrPath = 'Data/TFfiles/' + 'TFrTrain_' + NOWCLASS + ".tfrecords"
TrainingPart1TFrPath = 'Data/TFfiles/' + 'TFrTrainPart1_' + NOWCLASS + ".tfrecords"
TrainingPart2TFrPath = 'Data/TFfiles/' + 'TFrTrainPart2_' + NOWCLASS + ".tfrecords"
QuestTFrPath = 'Data/TFfiles/' + 'TFrQuest_' + NOWCLASS + ".tfrecords"

#%%
AttrKeyDim, AttrKeyList = Function.ReadAttrKeyInfoCsvFile(AttrKeyCsvFilePath)

TrainingTotalSampleNum, TrainingPart1SampleNum, TrainingPart2SampleNum = Function.ReadTrainingInfoCsvFile(TrainingInfoCsvFilePath)


QuestSubClassSampleNum, SubClassCodInLabel = Function.ReadQuestInfoCsvFile(QuestInfoCsvFilePath)

QuestImgPath, QuestAtt = Function.ReadQuestCsvFile(QuestCsvFilePath, SubClassCodInLabel)

#%% ----------------------------------------------------------Read TF files ------------------------------------------------------------------------------####


#-------------------------------------------------------------------------------------------------------------------------------------------
#ReadyTrainingTotalImageFlow, ReadyTrainingTotalLabelFlow = Function.ReadImgLabTF(TrainingTotalTFrPath, AttrKeyDim)
#OriTrainingTatolImages, OriTrainingTatolLabels = Function.ReadImgLabTF(TrainingTotalTFrPath, AttrKeyDim)
#ReadyTrainingTotalImageFlow, ReadyTrainingTotalLabelFlow = tf.train.batch([OriTrainingTatolImages, OriTrainingTatolLabels],
#                                                                          batch_size = TrainingTotalBatchSize,
#                                                                          capacity = TrainingTotalSampleNum)

#OriTrainingTatolImages, OriTrainingTatolLabels = Function.ReadImgLabTF(TrainingTotalTFrPath, AttrKeyDim)
#ReadyTrainingTotalImageFlow, ReadyTrainingTotalLabelFlow = tf.train.shuffle_batch([OriTrainingTatolImages, OriTrainingTatolLabels], 
#                                                  batch_size = TrainingTotalBatchSize, 
#                                                  capacity = TrainingTotalSampleNum, 
#                                                  min_after_dequeue=TrainingTotalSampleNum/2)


#-------------------------------------------------------------------------------------------------------------------------------------------
#TrainingPart1SampleNum = 1000
OriTrainingPart1Images, OriTrainingPart1Labels = Function.ReadImgLabTF(TrainingPart1TFrPath, AttrKeyDim)
ReadyTrainingPart1ImageFlow, ReadyTrainingPart1LabelFlow = tf.train.shuffle_batch([OriTrainingPart1Images, OriTrainingPart1Labels], 
                                                  batch_size = TrainingPart1BatchSize, 
                                                  capacity = TrainingPart1SampleNum, 
                                                  min_after_dequeue=TrainingPart1SampleNum/2)



#-------------------------------------------------------------------------------------------------------------------------------------------
#ReadyTrainingPart2ImageFlow, ReadyTrainingPart2LabelFlow = Function.ReadImgLabTF(TrainingPart2TFrPath, AttrKeyDim)
OriTrainingPart2Images, OriTrainingPart2Labels = Function.ReadImgLabTF(TrainingPart2TFrPath, AttrKeyDim)
ReadyTrainingPart2ImageFlow, ReadyTrainingPart2LabelFlow = tf.train.batch([OriTrainingPart2Images, OriTrainingPart2Labels],
                                                                                  batch_size = TrainingPart2BatchSize, 
                                                                                  capacity = TrainingPart2SampleNum)

#OriTrainingPart2Images, OriTrainingPart2Labels = Function.ReadImgLabTF(TrainingPart2TFrPath, AttrKeyDim)
#ReadyTrainingPart2ImageFlow, ReadyTrainingPart2LabelFlow = tf.train.shuffle_batch([OriTrainingPart2Images, OriTrainingPart2Labels], 
#                                                  batch_size = TrainingPart2BatchSize, 
#                                                  capacity = TrainingPart2SampleNum, 
#                                                  min_after_dequeue=TrainingPart2SampleNum/2)

#-------------------------------------------------------------------------------------------------------------------------------------------
ReadyQuestImageFlow = Function.ReadQuestTFr(QuestTFrPath)

#%%
################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

#%%
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


#%%

net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

#x = tf.placeholder(tf.float32, (None,) + xdim)

images = tf.reshape(ReadyTrainingPart1ImageFlow, shape=[-1, 227,227, 3]) # [batch, in_height, in_width, in_channels] 
#images = tf.reshape(ReadyTrainingTotalImageFlow, shape=[-1, 227,227, 3]) # [batch, in_height, in_width, in_channels] 

#x=tf.cast(images, tf.float32)
x=(tf.cast(images, tf.float32)/255.-0.5)*2#normalization
#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
#conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)

#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')

#fc7W = tf.Variable(net_data["fc7"][0])
#fc7b = tf.Variable(net_data["fc7"][1])
#fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

fc7W = tf.get_variable('fc7W', [4096,4096],initializer=tf.contrib.layers.xavier_initializer())
fc7b = tf.get_variable('fc7b', [4096],initializer=tf.contrib.layers.xavier_initializer())
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
#fc8W = tf.Variable(net_data["fc8"][0])
#fc8b = tf.Variable(net_data["fc8"][1])
#fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

fc8W = tf.get_variable('fc8W', [4096,AttrKeyDim],initializer=tf.contrib.layers.xavier_initializer())
fc8b = tf.get_variable('fc8b', [AttrKeyDim],initializer=tf.contrib.layers.xavier_initializer())
train_fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#%%
#x = tf.placeholder(tf.float32, (None,) + xdim)
test_images = tf.reshape(ReadyTrainingPart2ImageFlow, shape=[-1, 227,227, 3]) # [batch, in_height, in_width, in_channels] 

#x=tf.cast(images, tf.float32)
x1=(tf.cast(test_images, tf.float32)/255.-0.5)*2#normalization
#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1_in = conv(x1, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
#conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)

#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
#fc7W = tf.Variable(net_data["fc7"][0])
#fc7b = tf.Variable(net_data["fc7"][1])
#fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
#fc8W = tf.Variable(net_data["fc8"][0])
#fc8b = tf.Variable(net_data["fc8"][1])
#fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

test_fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#%%

#%%
#x = tf.placeholder(tf.float32, (None,) + xdim)
test_images = tf.reshape(ReadyQuestImageFlow, shape=[-1, 227,227, 3]) # [batch, in_height, in_width, in_channels] 

#x=tf.cast(images, tf.float32)
x2=(tf.cast(test_images, tf.float32)/255.-0.5)*2#normalization
#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1_in = conv(x2, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
#conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)

#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
#fc7W = tf.Variable(net_data["fc7"][0])
#fc7b = tf.Variable(net_data["fc7"][1])
#fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
#fc8W = tf.Variable(net_data["fc8"][0])
#fc8b = tf.Variable(net_data["fc8"][1])
#fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

quest_fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)






#%%
#prob
#softmax(name='prob'))

train_one_hot1 = tf.argmax(ReadyTrainingPart1LabelFlow, 1)
train_one_hot = tf.one_hot(train_one_hot1, AttrKeyDim)
train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=train_fc8, labels=ReadyTrainingPart1LabelFlow)

#train_one_hot1 = tf.argmax(ReadyTrainingTotalLabelFlow, 1)
#train_one_hot = tf.one_hot(train_one_hot1, AttrKeyDim)
#loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc8, labels=ReadyTrainingTotalLabelFlow)


#compute softmax softmax_cross_entropy_with_logits

train_optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(train_loss) 

train_hypothesis = tf.nn.softmax(train_fc8)
train_prediction = tf.argmax(train_hypothesis, 1)
train_correct_prediction = tf.equal(train_prediction, tf.argmax(train_one_hot, 1))
train_acc = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

#%%

test_one_hot1 = tf.argmax(ReadyTrainingPart2LabelFlow, 1)
test_one_hot = tf.one_hot(test_one_hot1, AttrKeyDim)

test_loss = tf.nn.softmax_cross_entropy_with_logits(logits=test_fc8, labels=ReadyTrainingPart2LabelFlow)
test_hypothesis = tf.nn.softmax(test_fc8)
test_prediction = tf.argmax(test_hypothesis, 1)
test_correct_prediction = tf.equal(test_prediction, tf.argmax(test_one_hot, 1))
test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

#%%

quest_hypothesis = tf.nn.softmax(quest_fc8)
quest_prediction = tf.argmax(quest_hypothesis, 1)





#%%
def train(): 
   

    opti=train_optimizer
#    feed = feed_dict={net.placeholders['img']: batch_image, net.placeholders['labels']: batch_label}

    
    
    PredictResultArray = np.zeros((QuestSubClassSampleNum, AttrKeyDim))
#    AccQue =Queue.Queue.(50)
    init=tf.global_variables_initializer()
    with tf.Session() as session: 

        with tf.device("/gpu:1"):

            session.run(init) 

            coord = tf.train.Coordinator() 

            threads = tf.train.start_queue_runners(coord=coord) 
#%%training          
            TrainAccArray=[]
            TrainLossArray = []

            max_iter = Epoch*(TrainingTotalSampleNum)/TrainingTotalBatchSize

            iter=0 

            if os.path.exists(os.path.join("model",'model.ckpt')) is True: 

                tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model.ckpt')) 

            while iter<max_iter: 

                train_loss_np, train_acc_np, _, train_image_np, train_label_np, train_hyp_np = session.run([train_loss, 
                                                                                                            train_acc,
                                                                                                            opti,
                                                                                                            ReadyTrainingPart1ImageFlow, 
                                                                                                            ReadyTrainingPart1LabelFlow,
                                                                                                            train_hypothesis])
                
#                temp = np.uint8(image_np[0,:,:,:].reshape(227,227,3))
#                figure = plt.figure()
#                plt.imshow(temp)
#                plt.show()
                TrainAccArray.append(train_acc_np)
                train_loss_avg = np.mean(train_loss_np)
                print "[ %d/%d ]"%(iter*100/max_iter,100),
                print 'Training Loss:' , train_loss_avg
                TrainLossArray.append(train_loss_avg)
                    
                iter+=1
            
#%% test
            
            max_iter = TrainingPart2SampleNum/TrainingPart2BatchSize
            print('Testing!!!')
            TestLossArray = []
            TestAccArray = []
            iter=0         
            while iter<max_iter: 
                [test_image_np, test_label_np, test_hypothesis_np, test_loss_np, test_acc_np] = session.run([ReadyTrainingPart2ImageFlow, 
                                                                                                ReadyTrainingPart2LabelFlow, 
                                                                                                test_hypothesis, 
                                                                                                test_loss, 
                                                                                                test_accuracy])
                loss_avg = np.mean(test_loss_np)
                TestLossArray.append(loss_avg)
                TestAccArray.append(test_acc_np)
#                print " %d/%d "%(iter*100/max_iter,100),
                print "Test Loss: ", loss_avg
                
                iter+=1

            print('TestAvgLoss: [ %f ]  TestAvgAcc: [ %f]', np.mean(TestAccArray))
            

            
#%% question
            print('Generating Answer!!!')
            max_iter = QuestSubClassSampleNum/QuestBatchSize
            iter=0         
            while iter<max_iter: 
                [quest_image_np, quest_hypothesis_np] = session.run([ReadyQuestImageFlow,
                                                                        quest_hypothesis])
                PredictResultArray[iter][:] = quest_hypothesis_np
#                temp.append(test_accuracy_np)
#                print('Test Accuracy:', test_accuracy_np)  

                if iter > (max_iter-11):
                    listtemp = list(quest_hypothesis_np[0])
                    j = listtemp.index(max(listtemp))
                    print "Class: %s   AttrKey: %s"%(NOWCLASS, AttrKeyList[j])
                    temp = np.uint8(quest_image_np[:,:,:].reshape(227,227,3))
                    figure = plt.figure()
                    plt.imshow(temp)
                    plt.show()
                
                
                iter+=1
                       
            coord.request_stop()#queue need be turned off, otherwise it will report errors
            coord.join(threads)
            
    session.close()
       
    return TrainLossArray, TrainAccArray, TestLossArray, TestAccArray, PredictResultArray

#def test(): 
#    
#    init = tf.global_variables_initializer()
#    sess = tf.Session()
#    sess.run(init)
#    
#    t = time.time()
#    output = sess.run(prob, feed_dict = {x:[im1,im2]})
#
##Output:
#
#
#for input_im_ind in range(output.shape[0]):
#    inds = argsort(output)[input_im_ind,:]
#    print("Image", input_im_ind)
#    for i in range(5):
#        print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])
#
#print(time.time()-t)

#%%
#if __name__ == '__main__':
TrainLossArray, TrainAccArray, TestLossArray, TestAccArray, PredictResultArray = train()

Function.WriteQuestResult('Results//', NOWCLASS, PredictResultArray, QuestImgPath, QuestAtt)

Function.WriteQuestResultAdd('Results//', NOWCLASS, PredictResultArray, QuestImgPath, QuestAtt)




