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
from trainer_pgh import Trainer

# external libraries
import tensorflow as tf
import cv2
import numpy as np
import datetime as dt
#-------------------------------------------

# setup data handler
DH = ImageDataHandler_forSFT()
DH.initFromCache('~dataHandlerCache.npy')
#DH.setDataDir('../../../data/training_defRenders')
#DH.setLabelsDir('../../../data/training_defRenders')
#DH.setDataExtension('.png')
#DH.setLabelsExtension('.npy')
#DH.setTrainFraction(0.2)
#DH.setValidationFraction(0.2)
#DH.buildDataHandlerCache()

print('total number of data points in the train set = ', DH.getTrainDataSize())

# initialize a Trainer using the Data Handler
T = Trainer()
T.setDataHandler(DH)

# training
T.setTrainBatchSize(4)

# setup unet
myUnet = UNetToTrainForSFT()
myUnet.setInitFromScratch(True)

# training loop
nEpochs = 5
costTable = [] #epoch, iteration, cost

def resizeImagesForUnet(imageStack, newWidth=480, newHeight=480):
    """
    imageStack is a 4D tensor (or numpy array) with format (n, h, w, c)
    """
    assert (len(imageStack.shape) == 4)
    resizedStack = []
    for nim in range(imageStack.shape[0]):
        resizedIm = cv2.resize(imageStack[nim,:,:,:], (newWidth, newHeight))
        resizedStack.append(resizedIm)
    resizedStack = np.array(resizedStack)
    return resizedStack

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
    myUnet.initializeMomentumOptimizer()
    myUnet.initializeAdamOptimizer()
    myUnet.initializePerVertexl2Cost()
    saver = tf.train.Saver()

    with SessionWithExitSave(saver=saver, exit_save_path='./tf-saves/_lastest.ckpt') as sess:    
    #with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        while (T.getTrainEpochCounter() < nEpochs):
            # get paths to next batch of data and labels
            dataPaths, labelPaths = T.getNextTrainBatch()
            
            # load the next batch of data and labels
            data = DH.loadData(dataPaths)
            data = resizeImagesForUnet(data)
            labels = DH.loadLabels(labelPaths)
         
            feed_dict = {myUnet.m_inputStack:data, myUnet.m_vertexLabels:labels}
            cost = sess.run(myUnet.m_l2PredictionCost, feed_dict=feed_dict)    
    
            sess.run(myUnet.m_minimizeL2CostMomentum, feed_dict=feed_dict)
            print('Epoch : ', T.getTrainEpochCounter(), \
                  ' Batch : ', T.getTrainBatchCounter(),\
                  '/', T.getNMaxTrainBatches(), ' cost : ', cost)      
            costTable.append([T.getTrainEpochCounter(), T.getTrainBatchCounter(), cost])
            #save_time = dt.datetime.now().strftime('%Y%m%d-%H.%M.%S')
            #saver.save(sess, './tf-saves/mnist-{save_time}.ckpt')
     
#---------------
## for testing labels
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#
#label0 = labels[0,:,:,:]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(label0[:,:,0], label0[:,:,1], label0[:,:,2])
#plt.show()
    
            
            
            
            
            
            
            