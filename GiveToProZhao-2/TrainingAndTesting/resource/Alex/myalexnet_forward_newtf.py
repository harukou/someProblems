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

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from matplotlib import pyplot as plt
from scipy.ndimage import filters
import Queue
from numpy import random
from PIL import Image  #
import tensorflow as tf


from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
CLASSNUM = 11
SAMPLENUM = 12433    #AmazonEasy=4453 AmazonAll= 12433
TESTNUM = 6000
BATCHSIZE = 600
DATASETNAME = "AmazonTrainingSet.tfrecords"

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

    img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)

    print img,label

    return img, label

#%%
    

ori_image, ori_label = read_and_decode(DATASETNAME)
batch_image, batch_label = tf.train.shuffle_batch([ori_image, ori_label], 
                                                  batch_size = BATCHSIZE, 
                                                  capacity = 12433, 
                                                  min_after_dequeue=10000)
    
#batch_image, batch_label = read_and_decode("AmazonTrainingSet.tfrecords")





#%%

################################################################################
#Read Image, and change to BGR

#im1 = (imread("quail227.JPEG")[:,:,:3]).astype(float32)
#im1 = im1 - mean(im1)
#im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#
#im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
#im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


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
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
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

net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

#x = tf.placeholder(tf.float32, (None,) + xdim)
images = tf.reshape(batch_image, shape=[-1, 227,227, 3]) # [batch, in_height, in_width, in_channels] 

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
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

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

fc8W = tf.get_variable('fc8W', [4096,CLASSNUM],initializer=tf.contrib.layers.xavier_initializer())
fc8b = tf.get_variable('fc8b', [CLASSNUM],initializer=tf.contrib.layers.xavier_initializer())
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#%%
#prob
#softmax(name='prob'))

y_one_hot = tf.one_hot(batch_label, CLASSNUM)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc8, labels=y_one_hot)
#compute softmax softmax_cross_entropy_with_logits

train_optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss) 


hypothesis = tf.nn.softmax(fc8)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#%%
def train(): 
   

    opti=train_optimizer
#    feed = feed_dict={net.placeholders['img']: batch_image, net.placeholders['labels']: batch_label}

    LossArray = []
    AccuracyArray=[]
    TotalAcc = 0
    
    
#    AccQue =Queue.Queue.(50)
    init=tf.global_variables_initializer()
    with tf.Session() as session: 

        with tf.device("/gpu:0"):

            session.run(init) 

            coord = tf.train.Coordinator() 

            threads = tf.train.start_queue_runners(coord=coord) 

            max_iter = (SAMPLENUM-TESTNUM)/BATCHSIZE

            iter=0 

            if os.path.exists(os.path.join("model",'model.ckpt')) is True: 

                tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model.ckpt')) 

            while iter<max_iter: 

                loss_np, _, image_np, label_np, inf_np, accuracy_np =session.run([loss, opti, batch_image, y_one_hot, fc8, accuracy]) 
                
#                temp = np.uint8(image_np[0,:,:,:].reshape(227,227,3))
#                figure = plt.figure()
#                plt.imshow(temp)
#                plt.show()
                
                loss_avg = np.mean(loss_np)
                AccuracyArray.append(accuracy_np)
                if iter % 10 == 0:
                    
                    print 'trainloss:' , accuracy_np
                    LossArray.append(loss_avg)
                    
                iter+=1
            
#            coord.request_stop()#queue need be turned off, otherwise it will report errors
#
#            coord.join(threads)
#            
#            
#            
#            
#            
#            coord = tf.train.Coordinator() 
#            threads = tf.train.start_queue_runners(coord=coord) 
            
            max_iter = TESTNUM/BATCHSIZE
            iter=0
            tempA = []
            
            while iter<max_iter: 
                image_np, label_np, result, Acc = session.run([batch_image, y_one_hot,prediction, accuracy])
                if iter%10 ==0:
                    print(Acc)
                tempA.append(Acc)
                iter+=1
            TotalAcc = np.mean(tempA)     
            
            coord.request_stop()#queue need be turned off, otherwise it will report errors
            coord.join(threads)
            
            
    session.close()
       
    return LossArray, AccuracyArray, TotalAcc
    
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
LossArray, AccuracyArray, TotalAcc = train()

