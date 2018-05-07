#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:04:17 2018

@author: dy Dayea
"""


# =============================================================================
# used for generating TFfiles
#
# =============================================================================

import sys

reload(sys)

sys.setdefaultencoding('utf8')

import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
#import matplotlib.pyplot as plt 
#import numpy as np







#%%
def SaveTF(ImgPath, classes, SaveTFPath, TFname):
    
    writer= tf.python_io.TFRecordWriter(SaveTFPath + TFname + ".tfrecords") #要生成的文件
    for index,name in enumerate(classes):
        class_path=ImgPath+name+'//'
        for img_name in os.listdir(class_path): 
            img_path=class_path+img_name #每一个图片的地址
            img=Image.open(img_path)
            img= img.resize((128,128))
            img_raw=img.tobytes()#将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])), #label: 1, 2, 3, ....
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  #序列化为字符串
        
    writer.close()


#%%
#read TF data
def ReadTF(path): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([path + '.tfrecords'])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  #reshape为128*128的3通道图片
#    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label


#%%
def SaveSomeImg(Img, Label, path):

    with tf.Session() as sess: #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(6):
            [example, l] = sess.run([Img,Label])#在会话中取出image和label
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            img.save(path+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            print(example, l)
        coord.request_stop()
        coord.join(threads)

    
#%%
    
DataPath='data//'
ImgSavePath = 'ImgOutPut//'
TFsavePath = 'TFOutPut//'
TFFileName = 'MyTF'
Classes={'cat',
         'dog'} #人为 设定 2类

SaveTF(ImgPath = DataPath, classes = Classes, SaveTFPath = TFsavePath, TFname = TFFileName)

ImgFlow, LabelFlow = ReadTF(TFsavePath + TFFileName)

SaveSomeImg(ImgFlow, LabelFlow, ImgSavePath) #have a test