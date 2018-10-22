#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:57:22 2018
@author: Pamir Ghimire
About : Trainer creates batches out of training, validation and testing data 
that is loaded from the disk by a dataHandler (check 'dataHandler_pgh.py')

# How to use:
# create a DataHandler
#tic = time.time()
#DH = ImageDataHandler_forSFT()
#DH.setDataDir('../../../data/training_defRenders')
#DH.setLabelsDir('../../../data/training_defRenders')
#DH.setDataExtension('.png')
#DH.setLabelsExtension('.npy')
#DH.buildDataHandlerCache()
#
#toc = time.time() - tic
#print('DataHandler : time elapsed in building cache =', toc-tic )
#
## initialize a Trainer using the Data Handler
#T = Trainer()
#T.setDataHandler(DH)
#
## training
#T.setTrainBatchSize(4)
#print('training batch counter : ', T.getTrainBatchCounter())
#tic = time.time()
#while (T.getTrainBatchCounter() < T.getNMaxTrainBatches()-1):
#    dataPaths, labelPaths = T.getNextTrainBatch()
#    data = DH.loadData(imagePaths=dataPaths, grayScale=False)
#    labels = DH.loadLabels(labelPaths=labelPaths)
#    print('batch counter : ',  T.getTrainBatchCounter())    
#
#toc = time.time()
#print('time taken to fetch one batch = ', (toc-tic)/T.getNMaxTrainBatches())  

"""
import numpy as np
import time

from dataHandler_pgh import DataHandler 
from dataHandler_pgh import ImageDataHandler_forSFT

class Trainer:
    """
    The Trainer is supposed to use the work done by DataHandler in surveying
    data and label files and generate batches of {data, label} for training a
    neural network
    
    This class helps to facilitate, for example, reshuffling of data after 
    every epoch for further training
    
    You can specify different batch sizes for validation, training and test
    splits of your data and simply query the Trainer for the 'next batch' of 
    data from one of these splits
    
    You can also query the 'current epoch' in one of the splits (validation, train
    test) after having asked the trainer for some 'next batch'es
    """
    def __init__(self):
        self.m_dataHandler = DataHandler()
            
        self.m_validationBatchSize = 1
        self.m_validationBatchCounter = 0
        self.m_validationEpochCounter = 0
        self.m_validationMaxBatches = -1
        self.m_validationShuffleIndx = []
        
        self.m_trainBatchSize = 1
        self.m_trainBatchCounter = 0
        self.m_trainEpochCounter = 0
        self.m_trainMaxBatches = -1
        self.m_trainShuffleIndx = []

        self.m_testBatchSize = 1
        self.m_testBatchCounter = 0
        self.m_testEpochCounter = 0
        self.m_testMaxBatches = -1
        self.m_testShuffleIndx = []
        
    def setDataHandler(self, dataHandler):
        """
        setDataHandler(dataHandler) : spefify the dataHandler that the trainer
        is to use, must be of type DataHandler, refer dataHandler_phg.py
        """
        assert(~dataHandler.isEmpty())
        self.m_dataHandler = dataHandler
        self.m_validationShuffleIndx = np.random.permutation(dataHandler.getValidationDataSize())
        self.m_trainShuffleIndx = np.random.permutation(dataHandler.getTrainDataSize())
        self.m_testShuffleIndx = np.random.permutation(dataHandler.getTestDataSize())
    
    def setTrainBatchSize(self, trainBatchSize):
        """
        setTrainBatchSize(trainBatchSize : int) : specify batch size of the 
        training split of the data
        """
        assert(trainBatchSize >= 1)
        assert(~self.m_dataHandler.isEmpty())
        assert(trainBatchSize <= self.m_dataHandler.getTrainDataSize())
        self.m_trainBatchSize = trainBatchSize
        
    def getTrainBatchSize(self):
        """
        getTrainBatchSize() : get batch size of the training split of the data
        """
        return self.m_trainBatchSize
    
    def setValidationBatchSize(self, validationBatchSize):
        """
        setValidationBatchSize(validationBatchSize : int) : specify batch size of the 
        validation split of the data
        """        
        assert (validationBatchSize >= 1)
        assert(~self.m_dataHandler.isEmpty())
        assert(validationBatchSize <= self.m_dataHandler.getValidationDataSize())
        self.m_validationBatchSize = validationBatchSize
        
    def getValidationBatchSize(self):
        """
        getValidationBatchSize() : get batch size of the validation split of the data
        """
        return self.m_validationBatchSize
    
    def setTestBatchSize(self, testBatchSize):
        """
        setTestBatchSize(testBatchSize : int) : specify batch size of the 
        test split of the data
        """
        assert(testBatchSize >= 1)
        assert(~self.m_dataHandler.isEmpty())
        assert(testBatchSize <= self.m_dataHandler.getTestDataSize())
        self.m_testBatchSize = testBatchSize
            
    def getTestBatchSize(self):
        """
        getTestBatchSize() : get batch size of the test split of the data
        """
        return self.m_testBatchSize
    
    def getNMaxValidationBatches(self):
        """
        getNMaxValidationBatches(): Maximum number of validation batches at
        specified validation batch size in one epoch
        """
        assert(~self.m_dataHandler.isEmpty())
        return int(np.floor(self.m_dataHandler.getValidationDataSize() / self.m_validationBatchSize) + 1)

    def getNMaxTrainBatches(self):
        """
        getNMaxTrainBatches(): Maximum number of train batches at
        specified train batch size in one epoch
        """
        assert(~self.m_dataHandler.isEmpty())
        return int(np.floor(self.m_dataHandler.getTrainDataSize() / self.m_trainBatchSize) + 1)

    def getNMaxTestBatches(self):
        """
        getNMaxTestBatches(): Maximum number of test batches at
        specified test batch size in one epoch
        """
        assert(~self.m_dataHandler.isEmpty())
        return int(np.floor(self.m_dataHandler.getTestDataSize() / self.m_testBatchSize) + 1)
    
    def getValidationBatchCounter(self):
        """
        getValidationBatchCounter() : get the number of validation batches fetched
        from the trainer in the current epoch
        """
        return self.m_validationBatchCounter
        
    def getTrainBatchCounter(self):
        """
        getTrainBatchCounter() : get the number of train batches fetched
        from the trainer in the current epoch
        """      
        return self.m_trainBatchCounter
        
    def getTestBatchCounter(self):
        """
        getTestBatchCounter() : get the number of test batches fetched
        from the trainer in the current epoch
        """   
        return self.m_testBatchCounter
        
    def getValidationEpochCounter(self):
        """
        getValidationEpochCounter() : get the number of validation epochs fetched
        from the Trainer 
        """  
        return self.m_validationEpochCounter
    
    def getTrainEpochCounter(self):
        """
        getTrainEpochCounter() : get the number of train epochs fetched
        from the Trainer        
        """
        return self.m_trainEpochCounter
    
    def getTestEpochCounter(self):
        """
        getTestEpochCounter() : get the number of test epochs fetched
        from the Trainer
        """
        return self.m_testEpochCounter
        
    def getNextValidationBatch(self):
        """
        getNextValidationBatch() : get the next batch of data and labels from the 
        validation split of the data in Trainer's DataHandler containing specified
        number of items in one validation batch (validationBatchSize)
        """
        assert(~self.m_dataHandler.isEmpty())
        startIndx = self.m_validationBatchSize * self.getValidationBatchCounter()
        endIndx = startIndx + self.getValidationBatchSize()
        indices = np.mod(range(startIndx, endIndx), self.m_dataHandler.getValidationDataSize())
        indices_sh = [self.m_validationShuffleIndx[i] for i in indices]
        data = [self.m_dataHandler.getDataHandlerCache()['validation']['data'][i] for i in indices_sh]
        labels = [self.m_dataHandler.getDataHandlerCache()['validation']['labels'][i] for i in indices_sh]
        
        self.m_validationBatchCounter += 1
        if self.m_validationBatchCounter >= self.m_dataHandler.getValidationDataSize():
            self.m_validationBatchCounter = np.mod(self.m_validationBatchCounter, self.m_dataHandler.getValidationDataSize())
            self.m_validationEpochCounter += 1
            self.m_validationShuffleIndx = np.random.permutation(self.m_dataHandler.getValidationDataSize())
        
        return data, labels
        
    def getNextTrainBatch(self):
        """
        getNextTrainBatch() : get the next batch of data and labels from the 
        train split of the data in Trainer's DataHandler containing specified
        number of items in one train batch (trainBatchSize)
        """
        assert(~self.m_dataHandler.isEmpty())
        startIndx = self.m_trainBatchSize * self.getTrainBatchCounter()
        endIndx = startIndx + self.getTrainBatchSize()
        indices = np.mod(range(startIndx, endIndx), self.m_dataHandler.getTrainDataSize())
        indices_sh = [self.m_trainShuffleIndx[i] for i in indices]
        data = [self.m_dataHandler.getDataHandlerCache()['train']['data'][i] for i in indices_sh]
        labels = [self.m_dataHandler.getDataHandlerCache()['train']['labels'][i] for i in indices_sh]
        
        self.m_trainBatchCounter += 1
        if self.m_trainBatchCounter >= self.m_dataHandler.getTrainDataSize():
            self.m_trainBatchCounter = np.mod(self.m_trainBatchCounter, self.m_dataHandler.getTrainDataSize())
            self.m_trainEpochCounter += 1
            self.m_trainShuffleIndx = np.random.permutation(self.m_dataHandler.getTrainDataSize())
        
        return data, labels
        
    def getNextTestBatch(self):
        """
        getNextTestBatch() : get the next batch of data and labels from the 
        test split of the data in Trainer's DataHandler containing specified
        number of items in one test batch (trainBatchSize)
        """       
        assert(~self.m_dataHandler.isEmpty())
        startIndx = self.m_testBatchSize * self.getTestBatchCounter()
        endIndx = startIndx + self.getTestBatchSize()
        indices = np.mod(range(startIndx, endIndx), self.m_dataHandler.getTestDataSize())
        indices_sh = [self.m_testShuffleIndx[i] for i in indices]
        data = [self.m_dataHandler.getDataHandlerCache()['test']['data'][i] for i in indices_sh]
        labels = [self.m_dataHandler.getDataHandlerCache()['test']['labels'][i] for i in indices_sh]
        
        self.m_testBatchCounter += 1
        if self.m_testBatchCounter >= self.m_dataHandler.getTestDataSize():
            self.m_testBatchCounter = np.mod(self.m_testBatchCounter, self.m_dataHandler.getTestDataSize())
            self.m_testEpochCounter += 1
            self.m_testShuffleIndx = np.random.permutation(self.m_dataHandler.getTestDataSize())
        
        return data, labels
        
#-----------------------------
## How to use:
## create a DataHandler
#tic = time.time()
#DH = ImageDataHandler_forSFT()
#DH.setDataDir('../../../data/training_defRenders')
#DH.setLabelsDir('../../../data/training_defRenders')
#DH.setDataExtension('.png')
#DH.setLabelsExtension('.npy')
#DH.buildDataHandlerCache()
#
#toc = time.time() - tic
#print('DataHandler : time elapsed in building cache =', toc-tic )
#
## initialize a Trainer using the Data Handler
#T = Trainer()
#T.setDataHandler(DH)
#
## training
#T.setTrainBatchSize(4)
#print('training batch counter : ', T.getTrainBatchCounter())
#tic = time.time()
#while (T.getTrainBatchCounter() < T.getNMaxTrainBatches()-1):
#    dataPaths, labelPaths = T.getNextTrainBatch()
#    data = DH.loadData(imagePaths=dataPaths, grayScale=False)
#    labels = DH.loadLabels(labelPaths=labelPaths)
#    print('batch counter : ',  T.getTrainBatchCounter())    
#
#toc = time.time()
#print('time taken to fetch one batch = ', (toc-tic)/T.getNMaxTrainBatches())  

