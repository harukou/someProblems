#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 21:41:00 2018

@author: dayea
"""

import sys

reload(sys)

sys.setdefaultencoding('utf8')

import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
#import matplotlib.pyplot as plt 
#import numpy as np
import MKTFfunction

#%%
class_name = ['collar_design_labels',
              'neckline_design_labels',
              'skirt_length_labels',
              'sleeve_length_labels',
              'neck_design_labels', 
              'coat_length_labels',
              'lapel_design_labels',
              'pant_length_labels']

AttrKeyCsvPath = "AttrKey//"
TrainImgBasePath = "TianChi/fashionAI_attributes_train_20180222/base//"
TrainLabCsvPath = "TianChi/fashionAI_attributes_train_20180222/base/Annotations/label.csv"
QuestImgBasePath = "TianChi/fashionAI_attributes_test_a_20180222/rank//"
QuestScvPath = "TianChi/fashionAI_attributes_test_a_20180222/rank/Tests/question.csv"


def op_data(StorePath, StorePath2, TrainImgBasePath, TrainLabCsvPath, QuestImgBasePath, QuestScvPath, NOWCLASS, AttrKeyCsvPath):
    
    #%% Read csv files
    AttrDim = MKTFfunction.ReadAttrCsv(AttrKeyCsvPath, NOWCLASS)
    
    TrainImgPath, TrainFeatureType, TrainLabels, AllTrainSampleNum = MKTFfunction.ReadLabelCsv(TrainLabCsvPath)
    
    QuestImgPath, QuestFeatureType, AllQuestSampleNum = MKTFfunction.ReadQuestLabelCsv(QuestScvPath)
    
    #%% 
    
    StartCod, EndCod, ThisClassSampleNum = MKTFfunction.FindClassNumForOneAtt(TrainFeatureType, NOWCLASS, AllTrainSampleNum)
    ThrSampleNum = StartCod + int(ThisClassSampleNum*0.85)
    SubClassCodInLabel = [StartCod, ThrSampleNum, EndCod]
    
    QuestStartCod, QuestEndCod, ThisQuestClassSampleNum = MKTFfunction.FindClassNumForOneAtt(QuestFeatureType, NOWCLASS, AllQuestSampleNum)
    
    SubQuestClassCodInLabel = [QuestStartCod, QuestEndCod]
    
    
    
    #%% Achieve encode labels
    
    SubClassLabels, SubClassSamoleNum = MKTFfunction.GenerateSubClassLabel(TrainLabels, ThisClassSampleNum, StartCod, EndCod, AttrDim)
    
    
    #%% Write csv files
    MKTFfunction.WriteTrainCsvFile(StorePath2, NOWCLASS, ThisClassSampleNum, AttrDim, SubClassCodInLabel)
    
    MKTFfunction.WriteQuestCsvFile(StorePath2, NOWCLASS, ThisQuestClassSampleNum, SubQuestClassCodInLabel)
    
    #%% Generate TF files
    
    SubClassTotalSampleNum = SubClassCodInLabel[2] - SubClassCodInLabel[0] + 1
    SubClassPart1SampleNum = SubClassCodInLabel[1] - SubClassCodInLabel[0] + 1
    SubClassPart2SampleNum = SubClassCodInLabel[2] - SubClassCodInLabel[1]
    QuestClassNum = QuestEndCod - QuestStartCod + 1
    
    MKTFfunction.GenerateImgLabTF(StorePath = StorePath,
                                  Name = 'TFrTrain_'+NOWCLASS,
                                  BasePath = TrainImgBasePath,
                                  ImgPathList = TrainImgPath[SubClassCodInLabel[0]: (SubClassCodInLabel[2] + 1)],
                                  LabList = SubClassLabels,
                                  SampleNum = SubClassTotalSampleNum)
    

    MKTFfunction.GenerateImgLabTF(StorePath = StorePath,
                                  Name = 'TFrTrainPart1_' + NOWCLASS,
                                  BasePath = TrainImgBasePath,
                                  ImgPathList = TrainImgPath[SubClassCodInLabel[0]: (SubClassCodInLabel[1] + 1)],
                                  LabList = SubClassLabels[0:(SubClassCodInLabel[1] - SubClassCodInLabel[0] + 1)],
                                  SampleNum = SubClassPart1SampleNum)
    
    MKTFfunction.GenerateImgLabTF(StorePath = StorePath,
                                  Name = 'TFrTrainPart2_'+NOWCLASS,
                                  BasePath = TrainImgBasePath,
                                  ImgPathList = TrainImgPath[(SubClassCodInLabel[1] + 1): (SubClassCodInLabel[2] + 1)],
                                  LabList = SubClassLabels[(SubClassCodInLabel[1] - SubClassCodInLabel[0] + 1):(SubClassCodInLabel[2]-SubClassCodInLabel[0]+1)],
                                  SampleNum = SubClassPart2SampleNum)
    
    MKTFfunction.GenerateImgTF(StorePath = StorePath,
                               Name = 'TFrQuest_'+NOWCLASS,
                               BasePath = QuestImgBasePath,
                               ImgPathList = QuestImgPath[QuestStartCod: (QuestEndCod + 1)],
                               SampleNum = QuestClassNum)
    

#%%
    
    
op_data('TFfiles//', 'CSVfiles//', TrainImgBasePath, TrainLabCsvPath, QuestImgBasePath, QuestScvPath, class_name[2], AttrKeyCsvPath)

for name in class_name:
    op_data('TFfiles//', 'CSVfiles//', TrainImgBasePath, TrainLabCsvPath, QuestImgBasePath, QuestScvPath, name, AttrKeyCsvPath)
    print name+'completed'
    


#%%
image, label = MKTFfunction.ReadImgLabTFfile(Name = ['TFrTrainPart1_'+class_name[0] + ".tfrecords"])



#%%
StorePath = "OutputFiles//"
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    
    for i in range(5):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(StorePath + str(i)+'_''Lab_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)
