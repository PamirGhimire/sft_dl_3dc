#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:19:11 2018

@author: Pamir Ghimire
About : This script trains 
"""
# paths to other necessary files
import sys
#sys.path.append('/home/bokoo/Desktop/sft_dl_3dc/src/networks/')
sys.path.append('../networks/')
sys.path.append('../dataHandling/')

from UNet_pgh import UNet
from UNet_pgh import UNetToTrainForSFT
from dataHandler_pgh import ImageDataHandler_forSFT
from UNet_pgh import resizeImagesForUnet
from trainer_pgh import Trainer

# external libraries
import tensorflow as tf
import cv2
import numpy as np
import datetime as dt
#-------------------------------------------
# params to control training
nEpochs = 5
trainBatchSize = 4
restoreCkptFileDir = ''#'./tf-saves/' # '' implies start training from scratch
restoreDataHandlerCache = './~dataHandlerCache.npy'
#-------------------------------------------

# setup data handler
DH = ImageDataHandler_forSFT()
#DH.initFromCache(restoreDataHandlerCache)
DH.setDataDir('../../../data/training_defRenders')
DH.setLabelsDir('../../../data/training_defRenders')
DH.setDataExtension('.png')
DH.setLabelsExtension('.npy')
DH.setTrainFraction(0.2)
DH.setValidationFraction(0.2)
DH.buildDataHandlerCache()

print('total number of data points in the train set = ', DH.getTrainDataSize())

# initialize a Trainer using the Data Handler
T = Trainer()
T.setDataHandler(DH)

# training
T.setTrainBatchSize(trainBatchSize)

# setup unet
myUnet = UNetToTrainForSFT()
myUnet.setInitFromScratch(True)

# training loop
costTable = [] #epoch, iteration, cost

#----------------------------
# for saving training progress at keyboard interrupt
class SessionWithExitSave(tf.Session):
    def __init__(self, *args, saver=None, exit_save_path=None, **kwargs):
        self.saver = saver
        self.exit_save_path = exit_save_path
        super().__init__(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            if self.saver:
                self.saver.save(self, self.exit_save_path)
                print('Output saved to: "{}./*"'.format(self.exit_save_path))
        super().__exit__(exc_type, exc_value, exc_tb)
#----------------------------
with tf.Graph().as_default():
    myUnet.initializeWeights()
    myUnet.initializeSgdOptimizer()
    myUnet.initializePerVertexl2Cost()
        
    saver = tf.train.Saver()

    with SessionWithExitSave(saver=saver, exit_save_path='./tf-saves/_lastest.ckpt') as sess:    
        if (restoreCkptFileDir != ''):    
            saver.restore(sess, tf.train.latest_checkpoint(restoreCkptFileDir))
        else:
            sess.run(tf.global_variables_initializer())
            
        allGlobalVars = tf.global_variables()
        allOps = tf.get_default_graph().get_operations()

        while (T.getTrainEpochCounter() < nEpochs):
            # get paths to next batch of data and labels
            dataPaths, labelPaths = T.getNextTrainBatch()
            
            # load the next batch of data and labels
            data = DH.loadData(dataPaths)
            data = resizeImagesForUnet(data)
            labels = DH.loadLabels(labelPaths)
         
            feed_dict = {myUnet.m_inputStack:data, myUnet.m_vertexLabels:labels}
            cost = sess.run(myUnet.m_l2PredictionCost, feed_dict=feed_dict)    
    
            sess.run(myUnet.m_minimizeL2CostSgd, feed_dict=feed_dict)
            print('Epoch : ', T.getTrainEpochCounter(), \
                  ' Batch : ', T.getTrainBatchCounter(),\
                  '/', T.getNMaxTrainBatches(), ' cost : ', cost)      
            costTable.append([T.getTrainEpochCounter(), T.getTrainBatchCounter(), cost])
            
            if (np.mod(T.getTrainBatchCounter(), 1000) == 0):
                save_time = dt.datetime.now().strftime('%Y%m%d-%H.%M.%S')
                saver.save(sess, './tf-saves/mnist-{save_time}.ckpt')
#     


            
            
            
            
            