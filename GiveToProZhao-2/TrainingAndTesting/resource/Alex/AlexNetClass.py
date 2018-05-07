#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 20:58:08 2018

@author: dayea
"""

import tensorflow as tf
import numpy as np

#tf.device(0)


#%%
def read_and_decode(filename):

    # generate quene according to file names
    # read tfrecords,return imgs and labels

    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)   #return file and name

    features = tf.parse_single_example(serialized_example,

                                       features={

                                           'label': tf.FixedLenFeature([], tf.int64),

                                           'img_raw' : tf.FixedLenFeature([], tf.string),

                                       })

 

    img = tf.decode_raw(features['img_raw'], tf.uint8)

    img = tf.reshape(img, [227, 227, 3])

 #    img = tf.reshape(img, [39, 39, 3])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)

    print img,label

    return img, label

#%%
class network(object): 
    
    def inference(self, images): 

        # transfer vectors to matrix  
        
        images = tf.reshape(images, shape=[-1, 227,227, 3]) # [batch, in_height, in_width, in_channels] 

        images=(tf.cast(images, tf.float32)/255.-0.5)*2#normalization


        #layer 1  define con bias and subsampling

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 4, 4, 1], padding='VALID'), 

                             self.biases['conv1']) 


        relu1= tf.nn.relu(conv1) 
#(1, 55, 55, 96) 
        pool1=tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 
#(1, 27, 27, 96)
 


        #layer 2

        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'), 

                             self.biases['conv2']) 

        relu2= tf.nn.relu(conv2) 

        pool2=tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 

 

 

        # layer3

        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'), 

                             self.biases['conv3']) 

        relu3= tf.nn.relu(conv3)

      #  pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

        conv4=tf.nn.bias_add(tf.nn.conv2d(relu3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'), 

                             self.biases['conv4']) 

        relu4= tf.nn.relu(conv4)

        conv5=tf.nn.bias_add(tf.nn.conv2d(relu4, self.weights['conv5'], strides=[1, 1, 1, 1], padding='SAME'), 

                             self.biases['conv5']) 

        relu5= tf.nn.relu(conv5)

        pool5=tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fc1,firstly transfer feture map to matrix

        flatten = tf.reshape(pool5, [-1, self.weights['fc1'].get_shape().as_list()[0]]) 

 

        drop1=tf.nn.dropout(flatten,0.5)

        fc1=tf.matmul(drop1, self.weights['fc1'])+self.biases['fc1'] 


        fc_relu1=tf.nn.relu(fc1)

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']

        fc_relu2=tf.nn.relu(fc2)

 

        fc3=tf.matmul(fc_relu2, self.weights['fc3'])+self.biases['fc3']


        return  fc3

 
#%%
    def __init__(self,net_data, class_num, Row, Cul,Chan): 

        self.class_num = class_num
        self.Row = Row
        self.Cul = Cul
        self.Chan = Chan
        #initate weight

        with tf.variable_scope("weights"):

           self.weights={

                #39*39*3->36*36*20->18*18*20 

                'conv1':tf.get_variable('conv1',[11,11,Chan,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

                #18*18*20->16*16*40->8*8*40 

                'conv2':tf.get_variable('conv2',[5,5,96,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

                #8*8*40->6*6*60->3*3*60 

                'conv3':tf.get_variable('conv3',[3,3,256,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

                #3*3*60->120 

                'conv4':tf.get_variable('conv4',[3,3,384,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 


                'conv5':tf.get_variable('conv5',[3,3,384,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 


                'fc1':tf.get_variable('fc1',[6*6*256,4096],initializer=tf.contrib.layers.xavier_initializer()), 


                'fc2':tf.get_variable('fc2',[4096,4096],initializer=tf.contrib.layers.xavier_initializer()), 


                #120->6 

                'fc3':tf.get_variable('fc3',[4096,class_num],initializer=tf.contrib.layers.xavier_initializer()), 

                } 

        with tf.variable_scope("biases"): 

            self.biases={ 

                'conv1':tf.get_variable('conv1',[96,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv2':tf.get_variable('conv2',[256,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv3':tf.get_variable('conv3',[384,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv4':tf.get_variable('conv4',[384,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'conv5':tf.get_variable('conv5',[256,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'fc1':tf.get_variable('fc1',[4096,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)), 

                'fc2':tf.get_variable('fc2',[4096,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'fc3':tf.get_variable('fc3',[class_num,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)) 

            }

       

#%%
    def inference_test(self,images): 

                # vector to matrix

                #used for test 

        images = tf.reshape(images, shape=[-1, self.Row,self.Cul,self.Chan])# [batch, in_height, in_width, in_channels] 
        images=(tf.cast(images,tf.float32)/255.-0.5)*2

 
        #layer 1

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 4, 4, 1], padding='VALID'), 
                             self.biases['conv1']) 
        #out (227-11)/4+1=54: [1, 227, 227, 3] -> [1,55,55,96]
        relu1= tf.nn.relu(conv1) 
        pool1=tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 
        #out (55-3)/2+1=27: [1,55,55,96] -> [1,27,27,96]
 
        #layer 2

        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 2, 2, 1], padding='VALID'), 
                             self.biases['conv2']) 
        #out (27-5)/1+1=12: [1, 27,27,96] -> [1,12,12,256]
        relu2= tf.nn.relu(conv2)

        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 
        #out (12-2)/2+1=6: [1,12,12,256] -> [1,6,6,256]

        # layer 3

#        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'), 
#                             self.biases['conv3']) 
#
#        relu3= tf.nn.relu(conv3) 
#        #out (6-3)/1+1=4: [1,6,6,256] -> [1,4,4,384]
#        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 
#        #out (4-2)/1+1=3: [1,4,4,384] -> [1,3,3,384]
 
        # FC1 transfer feture map to vector

        flatten = tf.reshape(pool2, [-1, self.weights['fc1'].get_shape().as_list()[0]]) 
        
        drop1=tf.nn.dropout(flatten,0.5)

        fc1=tf.matmul(drop1, self.weights['fc1'])+self.biases['fc1'] 

        fc_relu1=tf.nn.relu(fc1) 

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2'] 
        fc_relu2=tf.nn.relu(fc2)
        
        fc3=tf.matmul(fc_relu2, self.weights['fc3'])+self.biases['fc3'] 

        return  fc3

 
#%%
    #compute softmax softmax_cross_entropy_with_logits

    def sorfmax_loss(self, predicts, labels): 

        predicts=tf.nn.softmax(predicts) 
        y_one_hot = tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1])
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predicts, labels=y_one_hot)

      #  loss =-tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels) 

        self.cost= loss 

        return self.cost 

    #   
#%%
    def optimer(self,loss,lr=0.01): 

        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss) 

        return train_optimizer 


#%%

import  os 

def train(): 
    
    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    
   
    ori_image, ori_label = read_and_decode("AmazonTrainingSet.tfrecords")
    
    batch_image, batch_label = tf.train.shuffle_batch([ori_image, ori_label], 
                                                      batch_size = 30, 
                                                      capacity=2000, 
                                                      min_after_dequeue=1000)
   #net connection, used for trainning

    net=network(net_data=net_data, class_num=11, Row=227, Cul=227, Chan=3) #10 classes

    inf=net.inference_test(batch_image) 

    loss=net.sorfmax_loss(inf, batch_label) 

    opti=net.optimer(loss) 
#    feed = feed_dict={net.placeholders['img']: batch_image, net.placeholders['labels']: batch_label}

    LossArrsy = []

    init=tf.global_variables_initializer()

    with tf.Session() as session: 

        with tf.device("/gpu:0"):

            session.run(init) 

            coord = tf.train.Coordinator() 

            threads = tf.train.start_queue_runners(coord=coord) 

            max_iter = 13000

            iter=0 

            if os.path.exists(os.path.join("model",'model.ckpt')) is True: 

                tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model.ckpt')) 

            while iter<max_iter: 

                loss_np,_,label_np,image_np,inf_np=session.run([loss,opti,batch_image,batch_label,inf]) 
                loss_avg = np.mean(loss_np)
                if iter % 100 == 0:
                    
                    print 'trainloss:',loss_avg
                    LossArrsy.append(loss_avg)
                iter+=1 

            coord.request_stop()#queue need be turned off, otherwise it will report errors

            coord.join(threads)
    session.close()
    return LossArrsy




#%%
if __name__ == '__main__':
    LossArrsy = train()





