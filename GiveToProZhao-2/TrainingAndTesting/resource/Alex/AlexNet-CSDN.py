#!/usr/bin/env python2

# -*- coding: utf-8 -*-

"""

Created on Mon Jan 16 11:08:21 2017

@author: root

"""

import tensorflow as tf

#import frecordfortrain

tf.device(0)

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

   

def get_batch(image, label, batch_size,crop_size): 

        #inhance images

    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3]) #randomly cut part of Img

    distorted_image = tf.image.random_flip_up_down(distorted_image) #revolve updown and left-right

 

    #generate batch 

    #shuffle_batch's parameters:capacityis used to define shuttle's region,
    #If for the whole trainingdataset to get batch, then the capacity should be big  

    #make sure the dataset has been disorganie

    # num_threads=16,capacity=50000,min_after_dequeue=10000) 

    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size, 

                                                 num_threads=2,capacity=2,min_after_dequeue=10)

    # debuggig display

    #tf.image_summary('images', images) 

    print "in get batch"

    print images,label_batch

    return images, tf.reshape(label_batch, [batch_size])   


#from  data_encoder_decoeder import  encode_to_tfrecords,decode_from_tfrecords,get_batch,get_test_batch 

import  os 

class network(object): 

    def inference(self,images): 

        # transfer vectors to matrix  

        #images = tf.reshape(images, shape=[-1, 39,39, 3])

        images = tf.reshape(images, shape=[-1, 227,227, 3])# [batch, in_height, in_width, in_channels] 

        images=(tf.cast(images,tf.float32)/255.-0.5)*2#normalization





        #layer 1  define con bias and subsampling

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 4, 4, 1], padding='VALID'), 

                             self.biases['conv1']) 

 

        relu1= tf.nn.relu(conv1) 

        pool1=tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') 

 

 

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

 

    def __init__(self): 

        #initate weight

        with tf.variable_scope("weights"):

           self.weights={

                #39*39*3->36*36*20->18*18*20 

                'conv1':tf.get_variable('conv1',[11,11,3,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()), 

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

                'fc3':tf.get_variable('fc3',[4096,2],initializer=tf.contrib.layers.xavier_initializer()), 

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

                'fc3':tf.get_variable('fc3',[2,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)) 

 

            }

       


    def inference_test(self,images): 

                # vector to matrix

                #used for test 

        images = tf.reshape(images, shape=[-1, 39,39, 3])# [batch, in_height, in_width, in_channels] 

        images=(tf.cast(images,tf.float32)/255.-0.5)*2#normalization

 


        #layer 1

        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'), 

                             self.biases['conv1']) 

 

        relu1= tf.nn.relu(conv1) 

        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

 

 

        #layer 2

        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'), 

                             self.biases['conv2']) 

        relu2= tf.nn.relu(conv2) 

        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

 

 

        # layer 3

        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'), 

                             self.biases['conv3']) 

        relu3= tf.nn.relu(conv3) 

        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') 

 

 

        # FC1 transfer feture map to vector

        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]]) 

 

        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1'] 

        fc_relu1=tf.nn.relu(fc1) 

 

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2'] 

 

        return  fc2 

 

    #compute softmax softmax_cross_entropy_with_logits

    def sorfmax_loss(self,predicts,labels): 

        predicts=tf.nn.softmax(predicts) 

        labels=tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1]) 

        loss = tf.nn.softmax_cross_entropy_with_logits(predicts, labels)

      #  loss =-tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels) 

        self.cost= loss 

        return self.cost 

    #   

    def optimer(self,loss,lr=0.01): 

        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss) 

        return train_optimizer 


def train(): 

    batch_image, batch_label = read_and_decode("/home/zenggq/data/imagedata/data.tfrecords")

   #net connection, used for trainning

    net=network() 

    inf=net.inference(batch_image) 

    loss=net.sorfmax_loss(inf,batch_label) 

    opti=net.optimer(loss) 

    #test 

    init=tf.initialize_all_variables()

    with tf.Session() as session: 

        with tf.device("/gpu:1"):

            session.run(init) 

            coord = tf.train.Coordinator() 

            threads = tf.train.start_queue_runners(coord=coord) 

            max_iter = 9000 

            iter=0 

            if os.path.exists(os.path.join("model",'model.ckpt')) is True: 

                tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model.ckpt')) 

            while iter<max_iter: 

                loss_np,_,label_np,image_np,inf_np=session.run([loss,opti,batch_image,batch_label,inf]) 

                if iter%50==0: 

                    print 'trainloss:',loss_np 

                iter+=1 

            coord.request_stop()#queue need be turned off, otherwise it will report errors

            coord.join(threads)

 

if __name__ == '__main__':


    train()