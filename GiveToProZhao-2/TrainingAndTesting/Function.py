#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:52:47 2018

@author: scw4750
"""

import sys

reload(sys)

sys.setdefaultencoding('utf8')


import numpy as np
import tensorflow as tf 
import csv

#%%
def ReadAttrKeyInfoCsvFile(path):
    
    csvfile = open(path, 'r')
    print('用read_csv读取的csv文件：')
    csvreader = csv.reader(csvfile)
    
    AttrKeyList = []
    
    i = 0
    for line in csvreader:
        if i>0:
            AttrKeyList.append(line[0])
        
        i += 1
        
    csvfile.close()
    AttrKeyDim = i - 1
    
    return AttrKeyDim, AttrKeyList

#%%
def ReadTrainingInfoCsvFile(path):
    
    csvfile = open(path, 'r')
    print('用read_csv读取的csv文件：')
    csvreader = csv.reader(csvfile)
    
    i = 0
    for line in csvreader:
        if i == 0:
            TrainingTotalSampleNum = int(line[1])
        elif i == 2:
            SubClassCodInLabel = [int(line[1]), int(line[2]), int(line[3])]
        i += 1
    csvfile.close()
    
    TrainingPart1SampleNum = SubClassCodInLabel[1] - SubClassCodInLabel[0] + 1
    TrainingPart2SampleNum = SubClassCodInLabel[2] - SubClassCodInLabel[1] + 1
    return TrainingTotalSampleNum, TrainingPart1SampleNum, TrainingPart2SampleNum




#%%
def ReadQuestInfoCsvFile(path):
    
    csvfile = open(path,'r')
    print('用read_csv读取的csv文件：')
    csvreader = csv.reader(csvfile)
    
    QuestSubClassCodInLabel = []
    
    i = 0
    for line in csvreader:
        if i == 0:
            QuestSubClassSampleNum = int(line[1])
        elif i == 1:
            QuestSubClassCodInLabel = [int(line[1]), int(line[2])]
        i += 1
    csvfile.close()
    
    return QuestSubClassSampleNum, QuestSubClassCodInLabel


#%%
def ReadQuestCsvFile(path, QuestSubClassCodInLabel):
    
    csvfile = open(path,'r')
    print('用read_csv读取的csv文件：')
    csvreader = csv.reader(csvfile)
    
    
    QuestImgPath = []
    QuestAtt = []
    
    i = 0
    
    for line in csvreader:
        if i >= QuestSubClassCodInLabel[0] and i <= QuestSubClassCodInLabel[1]:
            QuestImgPath.append(line[0])
            QuestAtt.append(line[1])

        i += 1
        
    csvfile.close()
    
    return QuestImgPath, QuestAtt






#%%
def ReadImgLabTF(filename, labellength):

    # generate quene according to file names
    # read tfrecords,return imgs and labels

    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)   #return file and name

    features = tf.parse_single_example(serialized_example,

                                       features={
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([labellength], tf.float32),
                                               })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    
    label = tf.cast(features['label'], tf.int64)
#    label = tf.cast(features['label'], tf.int32)
    
    return img, label



#%%
def ReadQuestTFr(filename):

    # generate quene according to file names
    # read tfrecords, return imgs and labels

    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)   #return file and name

    features = tf.parse_single_example(serialized_example,

                                       features={
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                               })

    img = tf.decode_raw(features['img_raw'], tf.uint8)

    img = tf.reshape(img, [227, 227, 3])

    img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5

    return img










#%%
    
def WriteQuestResult(StorePath, NOWCLASS, PredictResultArray, QuestImgPath, QuestAtt):
    
    print('Writing Result: ' + NOWCLASS + '.csv文件：')

    CsvWriteFile = open(StorePath + 'result_part_for_' + NOWCLASS + '.csv', 'w')
    csvwriter = csv.writer(CsvWriteFile)
    
    for i in np.arange(len(QuestImgPath)):
        temp = []
        temp.append(QuestImgPath[i])
        temp.append(QuestAtt[i])
        
        value = ''
        for pro in PredictResultArray[i]:
            pro = '%.3f' %pro
            value = value + pro + ';'
            
        value = value[0:-1]
        temp.append(value)
        
        line = temp
        csvwriter.writerow(line)
    
    
    CsvWriteFile.close()
    print('Writing Result: ' + NOWCLASS + '.csv' + 'Completed!!!')
    

#%%
    
def WriteQuestResultAdd(StorePath, NOWCLASS, PredictResultArray, QuestImgPath, QuestAtt):
    
    print('Writing Result: ' + 'answer.csv文件：')

    CsvWriteFile = open(StorePath + 'answer' + '.csv', 'a+')
    csvwriter = csv.writer(CsvWriteFile)
    
    for i in np.arange(len(QuestImgPath)):
        temp = []
        temp.append(QuestImgPath[i])
        temp.append(QuestAtt[i])
        
        value = ''
        for pro in PredictResultArray[i]:
            pro = '%.3f' %pro
            value = value + pro + ';'
            
        value = value[0:-1]
        temp.append(value)
        
        line = temp
        csvwriter.writerow(line)
    
    
    CsvWriteFile.close()
    print('Writing Result: ' + NOWCLASS + '.csv' + 'Completed!!!')
    

