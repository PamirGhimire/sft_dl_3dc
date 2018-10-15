# -*- coding: utf-8 -*-
"""
A generic data handling class meant for supervised learning tasks, specifically
ones employing deep neural networks

Author : Pamir Ghimire, 14 October, 2018 

About :
    The class expects to find all your data to be in one directory
    and all the corresponding labels in another directory
    
    The class maintains a cache ('~dataHandlerCache.npy') in which it stores
    a table labelling each (data, label) pair as belonging to train, validation
    or test datasets (the dataset split information is stored as a dictionary,
    the key of the dictionary is full path of the file, the value is a list 
    with the following structure : ['train/test/validation'])
    
    You can use the setTrainFraction method to specify the desired split of 
    your dataset, along with setValidationFraction
    
    The validation set is a proper subset of the training set (usually 20%)
    
    If you have a datafile data0.dataExt in dataDir, in labelsDir, you must
    have data0.labelExt
    
    Example usage:
        # HOW TO USE DataHandler Class: 
        DH = DataHandler()
        DH.setDataDir('../../../data/training_defRenders')
        DH.setLabelsDir('../../../data/training_defRenders')
        DH.setDataExtension('.png') #or 'png'
        DH.setLabelsExtension('.npy') #or 'npy'
        DH.buildDataHandlerCache()
    
"""
# dependencies
import numpy as np
import os

class DataHandler:
    def __init__(self):
        # locations of data and labels
        self.m_dataDir = ''
        self.m_labelsDir = ''
        self.m_dataExtension = '' #ex : 'jpg', 'png', 'wav', 'raw'
        self.m_labelsExtension = '' #ex : 'txt', 'npy', 'npz', 'csv'
        
        # specifications about dataset split
        self.m_trainFraction = 0.8
        self.m_validationFraction = 0.2
        self.m_testFraction = 0.2
        assert(self.m_trainFraction + self.m_testFraction == 1.0)        

        # file to store information about dataset split
        self.m_dataHandlerCachePath = '~dataHandlerCache.npy'
        self.m_dataHandlerCache = dict(dict())
        if not (os.path.isfile(self.m_dataHandlerCachePath)):
            np.save(self.m_dataHandlerCachePath, self.m_dataHandlerCache)
            
    def isEmpty(self):
        if len(self.m_dataHandlerCache) <= 0:
            return True
        else:
            return False
        
    def setDataExtension(self, dataExt):
        if dataExt.startswith('.'):
            self.m_dataExtension = dataExt[1:]
        else:
            self.m_dataExtension = dataExt
        
    def getDataExtension(self):
        return self.m_dataExtension
    
    def setLabelsExtension(self, labelsExt):
        if labelsExt.startswith('.'):
            self.m_labelsExtension = labelsExt[1:]
        else:
            self.m_labelsExtension = labelsExt
        
    def getLabelsExtension(self):
        return self.m_labelsExtension
    
    def setDataDir(self, dataDir):
        assert(os.path.isdir(dataDir))
        self.m_dataDir = os.path.join(dataDir, "") #to add '/' or '\' 
        
    def getDataDir(self):
        return self.m_dataDir
    
    def setLabelsDir(self, labelsDir):
        assert(os.path.isdir(labelsDir))
        self.m_labelsDir = os.path.join(labelsDir, "")
        
    def getLabelsDir(self):
        return self.m_labelsDir
       
    def getDataHandlerCache(self):
        return self.m_dataHandlerCache

    def setTrainFraction(self, trainFraction):
        assert(trainFraction >= 0.0 and trainFraction <= 1.0)
        self.m_trainFraction = trainFraction
        self.m_testFraction = 1.0 - trainFraction
        assert(self.m_trainFraction + self.m_testFraction == 1.0)
        
    def getTrainFraction(self):
        return self.m_trainFraction
        
    def setValidationFraction(self, validationFraction):
        assert(validationFraction >= 0.0 and validationFraction <= 1.0)
        self.m_validationFraction = validationFraction
        
    def getValidationFraction(self):
        return self.m_validationFraction
    
    def setTestFraction(self, testFraction):
        assert(testFraction >= 0.0 and testFraction <= 1.0)
        self.m_testFraction =  testFraction
        self.m_trainFraction = 1.0 - testFraction
        assert(self.m_trainFraction + self.m_testFraction == 1.0)
        
    def getTestFraction(self):
        return self.m_testFraction
    
    # helper function
    def getFileNameWoExt(self, fn):
        if os.path.splitext(fn)[1] == '':
            return os.path.splitext(fn)[0]
        else:
            return self.getFileNameWoExt(os.path.splitext(fn)[0])

    def getValidationDataSize(self):
        assert(len(self.m_dataHandlerCache) > 0)
        return (len(self.m_dataHandlerCache['validation']['data']))
    
    def getTrainDataSize(self):
        assert(len(self.m_dataHandlerCache) > 0)
        return (len(self.m_dataHandlerCache['train']['data']))

    def getTestDataSize(self):
        assert(len(self.m_dataHandlerCache) > 0)
        return (len(self.m_dataHandlerCache['test']['data']))

    def buildDataHandlerCache(self):
       print('DataHandler : Building Cache ...')
       # some--sanity--checks----------
       assert(self.m_dataDir != '' and self.m_labelsDir != '')
       assert(self.m_dataExtension != '' and self.m_labelsExtension != '')
       assert(os.path.exists(self.m_dataDir) and len(os.listdir(self.m_dataDir)) > 0)
       assert(os.path.exists(self.m_labelsDir) and len(os.listdir(self.m_labelsDir)) > 0)
       assert(self.m_trainFraction >= 0 and self.m_trainFraction <= 1.0)
       assert(self.m_testFraction >= 0 and self.m_testFraction <= 1.0)
       assert(self.m_validationFraction >= 0 and self.m_validationFraction <= 1.0)
       assert(self.m_trainFraction + self.m_testFraction == 1.0)
       #----sanity--checks--end----------
       
       # list all data and label files
       allFilesInDataDir = os.listdir(self.m_dataDir)
       allDataFiles = []
       allLabelFiles = []
       for dataFile in allFilesInDataDir:
           if dataFile.endswith(self.m_dataExtension):
               allDataFiles.append(dataFile)
               labelFile = self.getFileNameWoExt(dataFile) + '.' + self.m_labelsExtension
               assert(os.path.isfile(os.path.join(self.m_labelsDir, labelFile)))
               allLabelFiles.append(labelFile)

       assert(len(allDataFiles) > 0) #at least one data file
       assert(len(allDataFiles) == len(allLabelFiles)) # label for each data
       
       # randomly choose which files should be used for training and testing
       shuffleIndx = np.random.permutation(len(allDataFiles))
       trainIndx = np.floor(self.m_trainFraction * len(allDataFiles))
       validationIndx = np.floor(self.m_validationFraction * trainIndx)
       
       # build the cache
       self.m_dataHandlerCache = dict(dict(dict()))
       trainData = []
       validationData = []
       testData = []
       trainLabels = []
       validationLabels = []
       testLabels = []
       for nFile in range(len(allDataFiles)):
           if (nFile < validationIndx):
               validationData.append(os.path.join( self.m_dataDir, allDataFiles[ shuffleIndx[nFile]]))
               validationLabels.append(os.path.join( self.m_labelsDir, allLabelFiles[ shuffleIndx[nFile]]))
           elif (nFile < trainIndx):
               trainData.append(os.path.join( self.m_dataDir, allDataFiles[ shuffleIndx[nFile]]))
               trainLabels.append(os.path.join( self.m_labelsDir, allLabelFiles[ shuffleIndx[nFile]]))
           else:
               testData.append(os.path.join( self.m_dataDir, allDataFiles[ shuffleIndx[nFile]]))
               testLabels.append(os.path.join( self.m_labelsDir, allLabelFiles[ shuffleIndx[nFile]]))
       # save the cache
       dictValidation = {'data':validationData, 'labels':validationLabels}
       dictTrain = {'data':trainData, 'labels':trainLabels}
       dictTest = {'data':testData, 'labels':testLabels}
       self.m_dataHandlerCache = {'validation':dictValidation, 'train':dictTrain, 'test':dictTest}
       np.save('~dataHandlerCache.npy', self.m_dataHandlerCache)
   

#---------------------------
# HOW TO USE DataHandler Class: 
#DH = DataHandler()
#DH.setDataDir('../../../data/training_defRenders')
#DH.setLabelsDir('../../../data/training_defRenders')
#DH.setDataExtension('.png')
#DH.setLabelsExtension('.npy')
#DH.buildDataHandlerCache()

#---------------------------
# ImageDataHandler class : DataHandler class specialized for handling image data
import cv2
       
class ImageDataHandler(DataHandler):
    def __init__(self):
        DataHandler.__init__(self);
        
    def loadData(self, imagePaths, grayScale=False):
       """
       imagePaths is a list of strings specifying paths to images to be loaded
       """
       assert(type(imagePaths) == list or type(imagePaths) == str)
       
       if type(imagePaths) == list:
           sampleIm = cv2.imread(imagePaths[0])
           w = cv2.shape[1]
           h = cv2.shape[0]
           ch = cv2.shape[3]
           d = len(imagePaths)
           if not grayScale:
               images = np.zeros((d, h, w, ch))
           else:
               images = np.zeros((d, h, w))
               
           nIm = 0
           for imagePath in imagePaths:
               if not grayScale:
                   image = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                   images[nIm,:,:] = image
               else:
                   image = cv2.imread(imagePath)
                   images[nIm,:,:,:] = image
       elif type(imagePaths) == str:
           if not grayScale:
               return cv2.imread(imagePath)
           else:
               return cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
   
#--------------------------------
# specialization of the ImageDataHandler class for handling SFT problem, when
# labels are 3D coordinates of vertices of the template mesh
class ImageDataHandler_forSFT(ImageDataHandler):
    def __init__(self):
        ImageDataHandler.__init__(self)
        
    def loadLabels(self, labelPaths):
        assert(type(labelPaths) == list and len(labelPaths) > 0 and labelPaths[0].endswith('.npy'))
        for labelPath in labelPaths:
            rendData = np.load(labelPath).item()
            vertices = rendData['mesh']['vertices'] 
            intrinsics = rendData['cameraMatrix']
            
        
        