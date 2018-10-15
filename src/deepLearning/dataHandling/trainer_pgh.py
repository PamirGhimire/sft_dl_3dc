#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:57:22 2018
@author: Pamir Ghimire
About : Trainer creates batches out of training, validation and testing data 
that is loaded from the disk by a dataHandler (check 'dataHandler_pgh.py')
"""
import numpy as np
from dataHandler_pgh import DataHandler

class Trainer:
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
        assert(~dataHandler.isEmpty())
        self.m_dataHandler = dataHandler
        self.m_validationShuffleIndx = np.random.permutation(dataHandler.getValidationDataSize())
        self.m_trainShuffleIndx = np.random.permutation(dataHandler.getTrainDataSize())
        self.m_testShuffleIndx = np.random.permutation(dataHandler.getTestDataSize())
    
    def setTrainBatchSize(self, trainBatchSize):
        assert(trainBatchSize >= 1)
        assert(~self.m_dataHandler.isEmpty())
        assert(trainBatchSize <= self.m_dataHandler.getTrainDataSize())
        self.m_trainBatchSize = trainBatchSize
        
    def getTrainBatchSize(self):
        return self.m_trainBatchSize
    
    def setValidationBatchSize(self, validationBatchSize):
        assert (validationBatchSize >= 1)
        assert(~self.m_dataHandler.isEmpty())
        assert(validationBatchSize <= self.m_dataHandler.getValidationDataSize())
        self.m_validationBatchSize = validationBatchSize
        
    def getValidationBatchSize(self):
        return self.m_validationBatchSize
    
    def setTestBatchSize(self, testBatchSize):
        assert(testBatchSize >= 1)
        assert(~self.m_dataHandler.isEmpty())
        assert(testBatchSize <= self.m_dataHandler.getTestDataSize())
        self.m_testBatchSize = testBatchSize
            
    def getTestBatchSize(self):
        return self.m_testBatchSize
    
    def getNMaxValidationBatches(self):
        assert(~self.m_dataHandler.isEmpty())
        return int(np.floor(self.m_dataHandler.getValidationDataSize() / self.m_validationBatchSize) + 1)

    def getNMaxTrainBatches(self):
        assert(~self.m_dataHandler.isEmpty())
        return int(np.floor(self.m_dataHandler.getTrainDataSize() / self.m_trainBatchSize) + 1)

    def getNMaxTestBatches(self):
        assert(~self.m_dataHandler.isEmpty())
        return int(np.floor(self.m_dataHandler.getTestDataSize() / self.m_testBatchSize) + 1)
    
    def getValidationBatchCounter(self):
        return self.m_validationBatchCounter
        
    def getTrainBatchCounter(self):
        return self.m_trainBatchCounter
        
    def getTestBatchCounter(self):
        return self.m_testBatchCounter
        
    def getValidationEpochCounter(self):
        return self.m_validationEpochCounter
    
    def getTrainEpochCounter(self):
        return self.m_trainEpochCounter
    
    def getTestEpochCounter(self):
        return self.m_testEpochCounter
        
    def getNextValidationBatch(self):
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
# How to use:
# create a DataHandler
DH = DataHandler()
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
print('training batch counter : ', T.getTrainBatchCounter())
#while (T.getTrainEpochCounter() < 3):
#    data, labels = T.getNextTrainBatch()
#    print('batch : ', T.getTrainBatchCounter(), '|| epoch : ', T.getTrainEpochCounter())
#    print(data[0:5])
#    print('--------')
#    print(labels[0:5])
#    print('---------------------------------------')
#    
## validation
#while (T.getValidationEpochCounter() < 3):
#    data, labels = T.getNextValidationBatch()
#    print('validation batch : ', T.getValidationBatchCounter(), '|| validation epoch : ', T.getValidationEpochCounter())
#    print(data[0:5])
#    print('--------')
#    print(labels[0:5])
#    print('---------------------------------------')
#    
# testing
T.setTestBatchSize(2)
while (T.getTestEpochCounter() < 3):
    data, labels = T.getNextTestBatch()
    print('test batch : ', T.getTestBatchCounter(), '|| test epoch : ', T.getTestEpochCounter())
    print(data[0:5])
    print('--------')
    print(labels[0:5])
    print('---------------------------------------')
    

