#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:19:11 2018

@author: Pamir Ghimire
About : This script trains 
"""
# paths to other necessary files
import sys
sys.path.append('../networks/')
sys.path.append('../dataHandling/')

from networks.UNet_pgh import UNet
from dataHandling.dataHandler_pgh import ImageDataHandler_forSFT
from dataHandling.trainer_pgh import Trainer
#-------------------------------------------

DH = ImageDataHandler_forSFT()
DH.setDataDir('../../../data/training_defRenders')
DH.setLabelsDir('../../../data/training_defRenders')
DH.setDataExtension('.png')
DH.setLabelsExtension('.npy')
DH.buildDataHandlerCache()

# initialize a Trainer using the Data Handler
T = Trainer()
T.setDataHandler(DH)

# training
T.setTrainBatchSize(4)

# training loop
nEpochs = 5

while (T.getTrainEpochCounter < nEpochs):
    # get paths to next batch of data and labels
    dataPaths, labelPaths = T.getNextTrainBatch()
    
    # load the next batch of data and labels
    data = DH.loadData(dataPaths)
    labels = DH.loadLabels(labelPaths)
    
    # feed the data to the network
    #feed_dict = {'data':data, 'labels':labels}
    
    # run a training step
    # note the training loss
    # estimate loss on validation set, note validation loss
    # if loss is not decreasing (or has platoed and started increasing)
        # terminate training
    