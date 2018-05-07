#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:00:49 2018

@author: scw4750
"""

import sys

reload(sys)

sys.setdefaultencoding('utf8')

import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
#import matplotlib.pyplot as plt 
import numpy as np
import csv



#%%
def ReadAttrCsv(LabelCsvPath, NOWCLASS):
    csvfile = open(LabelCsvPath + NOWCLASS +'_class.csv', 'r')
    print('读取的.csv文件：')
    csvreader = csv.reader(csvfile)
    
    AttrValue = []
    AllSampleNum = 0
    i = 0
    for line in csvreader:
        if i > 0:
            AttrValue.append(line[0])
            AllSampleNum += 1
        i += 1
        
    csvfile.close()
    AttrDim = len(AttrValue)
    return AttrDim


#%%
def ReadLabelCsv(LabelCsvPath):
    csvfile = open(LabelCsvPath, 'r')
    print('读取的label.csv文件：')
    csvreader = csv.reader(csvfile)
    
    TrainImgPath = []
    TrainFeatureType = []
    TrainLabels = []
    AllSampleNum = 0
    for line in csvreader:
        TrainImgPath.append(line[0])
        TrainFeatureType.append(line[1])
        TrainLabels.append(line[2])
        AllSampleNum += 1
    
    csvfile.close()
    return TrainImgPath, TrainFeatureType, TrainLabels, AllSampleNum


#%%
def ReadQuestLabelCsv(QuestCsvPath):
    
    csvfile = open(QuestCsvPath, 'r')
    print('读取的question.csv文件：')
    csvreader = csv.reader(csvfile)
    
    TrainImgPath = []
    TrainFeatureType = []

    AllSampleNum = 0
    for line in csvreader:
        TrainImgPath.append(line[0])
        TrainFeatureType.append(line[1])
        AllSampleNum += 1
    
    csvfile.close()
    return TrainImgPath, TrainFeatureType, AllSampleNum

#%% 

def FindClassNumForOneAtt(TrainFeatureType, NOWCLASS, AllSampleNum):

    StartCod = 0
    EndCod = 0
    FindStartFlag = 0
    FindEndFlag = 0
    
    for Index in range(AllSampleNum):
        if FindStartFlag == 0 and TrainFeatureType[Index] == NOWCLASS:
            StartCod = Index
            FindStartFlag = 1
        if FindStartFlag == 1 and (TrainFeatureType[Index] != NOWCLASS or Index == AllSampleNum - 1):
            
            EndCod = Index - 1
            if Index == AllSampleNum - 1:
                EndCod = EndCod + 1
            FindEndFlag = 1
        if FindEndFlag ==  1:
            break;
    
    ThisClassSampleNum = EndCod - StartCod + 1
    
    return StartCod, EndCod, ThisClassSampleNum

#%%
def GenerateSubClassLabel(TrainLabels, ThisClassSampleNum, StartCod, EndCod, AttrDim):
    
    SubClassSampleNum = 0
    SubClassLabels = np.zeros((ThisClassSampleNum, AttrDim), dtype = np.float32)
    
    for lab in TrainLabels[StartCod:EndCod + 1]:
        temp = np.zeros((AttrDim))
        i = 0
        for v in lab:
            if v == 'y':
                temp[i] = 1
            elif v== 'm':
                temp[i] = 0.6
            
            i += 1
        
        SubClassLabels[SubClassSampleNum] = temp
        SubClassSampleNum += 1    
        
    return SubClassLabels, SubClassSampleNum

#%%
def WriteTrainCsvFile(StorePath, NOWCLASS, ThisClassSampleNum, AttrDim, SubClassCodInLabel):
    print('Writing Train ' + NOWCLASS + '.csv文件：')

    CsvWriteFile = open(StorePath + 'Train_Info_for_' + NOWCLASS + '.csv', 'w')
    csvwriter = csv.writer(CsvWriteFile)
    
    line = ['ClassSampleNum_'+NOWCLASS, ThisClassSampleNum]
    csvwriter.writerow(line)
    
    line = ['AttrDim_' + NOWCLASS, AttrDim]
    csvwriter.writerow(line)
    
    line = ['SubClassCodInLabel', SubClassCodInLabel[0], SubClassCodInLabel[1], SubClassCodInLabel[2]]
    csvwriter.writerow(line)
    
#    for i in np.arange(len(SubClassList)):
#        line = [i, SubClassList[i]]
#        csvwriter.writerow(line)
    
    CsvWriteFile.close()
    

#%%
def WriteQuestCsvFile(StorePath, NOWCLASS, ThisClassSampleNum, SubClassCodInLabel):

    print('Writing question ' + NOWCLASS + '.csv文件：')
    CsvWriteFile = open(StorePath + 'Quest_Info_for_' + NOWCLASS + '.csv', 'w')
    csvwriter = csv.writer(CsvWriteFile)
    
    line = ['ClassSampleNum_'+NOWCLASS, ThisClassSampleNum]
    csvwriter.writerow(line)
    
    line = ['SubClassCodInLabel', SubClassCodInLabel[0], SubClassCodInLabel[1]]
    csvwriter.writerow(line)
    
    CsvWriteFile.close()


    #%%

def _int64_feture(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value = value))

def _float_feture(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value = value))


tf.train.Feature()

#%%
def GenerateImgLabTF(StorePath, Name, BasePath, ImgPathList, LabList, SampleNum):
    
    writer= tf.python_io.TFRecordWriter(StorePath + Name + ".tfrecords") #要生成的文件
    print('Generating TF file: ' + Name)
    for index in np.arange(SampleNum):
        img_path = BasePath + ImgPathList[index] #每一个图片的地址
        img=Image.open(img_path)
        img= img.resize((227,227))
        img_raw=img.tobytes()#将图片转化为二进制格式
        Lab = [float(i) for i in LabList[index]]
        example = tf.train.Example(features=tf.train.Features(feature={
             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
             "label": _float_feture(Lab)
        }))
        #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()



#%%
def GenerateImgTF(StorePath, Name, BasePath, ImgPathList, SampleNum):
    writer= tf.python_io.TFRecordWriter(StorePath + Name + ".tfrecords") #要生成的文件
    print('Generating TF file: ' + Name)
    for index in np.arange(SampleNum):
        img_path = BasePath + ImgPathList[index] #每一个图片的地址
        img=Image.open(img_path)
        img= img.resize((227,227))
        img_raw=img.tobytes()#将图片转化为二进制格式
        
        example = tf.train.Example(features=tf.train.Features(feature={
             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()





#%%
#read TF data
def ReadImgLabTF(filename, NormFlag = 0): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])  #reshape为128*128的3通道图片
    if NormFlag:
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    else:
        img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    
    return img, label




#%%
#read TF data
def ReadImgTF(filename, NormFlag = 0): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])  #reshape为128*128的3通道图片
    if NormFlag:
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    else:
        img = tf.cast(img, tf.float32)
        
    return img


#%%
def ReadImgLabTFfile(Name):
    
    filename_queue = tf.train.string_input_producer(Name) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 3])
    label = tf.cast(features['label'], tf.int32)
    
    return image, label
