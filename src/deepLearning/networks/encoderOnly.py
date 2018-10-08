# -*- coding: utf-8 -*-
"""
Created on Sun Apr  24 2018

@author: pgh
"""
import inspect
import os

import numpy as np
import tensorflow as tf
import time
import cv2

VGG_MEAN = [103.939, 116.779, 123.68]

class EncoderOnlySftNet:
    def __init__(self, weightsFile, resumeTraining=False):
        self.data_dict = np.load(weightsFile).item()
        self.imageWidth = 640
        self.imageHeight = 480
        self.allVars = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

        tf.reset_default_graph()
        self.batchSize = tf.placeholder(tf.int32, name='batchSize')
        
        # trainer: optimizer related settings
        #global_step = tf.Variable(0, trainable=False)
        #starter_learning_rate = 0.005
        #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
        #self.trainer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
        #self.trainer = tf.train.GradientDescentOptimizer(0.001)        

        # inputs
        self.inputImageStack = tf.placeholder(tf.float32, shape=(None, self.imageHeight, self.imageWidth, 1), name='anchor')

        # embeddings and loss minimization
#        self.anchorEmbs, self.posEmbs, self.negEmbs = self.getEmbeddings(self.anchor, self.positive, self.negative)
#        self.batchTripletLoss, self.posDists, self.negDists, self.posNegDists = self.tripletLoss(self.anchorEmbs, self.posEmbs, self.negEmbs)
#        self.trainingOp_tripletLoss = self.trainer.minimize(self.batchTripletLoss)
        self.convFeatures = vggEncoder(inputImageStack)

    #----------------------------------------------------------------------------
    def vggEncoder(self, x):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        # Convert RGB to BGR
        conv1_1 = self.conv_layer(x, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')
        
        trainable = True
        
        conv5_1 = self.conv_layer(pool4, "conv5_1", trainable=trainable)
        conv5_2 = self.conv_layer(conv5_1, "conv5_2", trainable=trainable)
        conv5_3 = self.conv_layer(conv5_2, "conv5_3", trainable=trainable)
        pool5 = self.max_pool(conv5_3, 'pool5')
        
        return pool5


    #----------------------------------------------------------------------------
    def conv_layer(self, bottom, name, trainable=False):
        filt = self.get_conv_filter(name, trainable)
        conv_biases = self.get_bias(name, trainable)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

        return fc

    #----------------------------------------------------------------------------
    def fcLayer(self, fcInput, fcWeights, fcBiases, name=None):
        shape = fcInput.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(fcInput, [-1, dim])

        fc = tf.nn.bias_add(tf.matmul(x, fcWeights), fcBiases, name=name)
        return fc

    #----------------------------------------------------------------------------
    def get_conv_filter(self, name, trainable=False):
        return tf.get_variable(name=name, initializer=tf.constant(self.data_dict[name]), trainable=trainable)
        #return tf.constant(self.data_dict[name][0], name="filter")

    #----------------------------------------------------------------------------
    def get_bias(self, name, trainable=False):
        return tf.get_variable(name=name+'b', initializer=tf.constant(self.data_dict[name+'b']), trainable=trainable)
        #return tf.constant(self.data_dict[name][1], name="biases")
  
    #----------------------------------------------------------------------------     
    def get_fc_weight(self, name, trainable=False):
        return tf.get_variable(name=name, initializer=tf.constant(self.data_dict[name][0]), trainable=trainable)
        #return tf.constant(self.data_dict[name][0], name="weights")
    
    #----------------------------------------------------------------------------
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #----------------------------------------------------------------------------    
    def saveWeights(self, weightsFileName):
           
        # save all variables in the chosen scope
        dict2save = {}
        with tf.variable_scope('tripletTower', reuse=True):
            for somevar in self.allVars:
                if not somevar == 'fcEmb':
                    dict2save[somevar] = tf.get_variable(somevar).eval()
                    dict2save[somevar+'b'] = tf.get_variable(somevar+'b').eval()
                else:
                    dict2save[somevar] = tf.get_variable(somevar).eval()
        
        print('\nSaving shared weights as : ', weightsFileName, '...\n')
        np.save(weightsFileName, dict2save)     
    
#------------------------------------------------------------------------------
#vggWeights = r'C:\Users\pgh\Documents\MachineLearning\python\1cnnFeaturesOffTheShelf\vgg16.npy' 
#mysiam = siamVgg(vggWeights, 10)
