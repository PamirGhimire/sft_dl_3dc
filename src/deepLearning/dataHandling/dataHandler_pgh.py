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
    """
    A generic class to deal with data and labels stored in a directory so that
    it is easy to use for training a neural network
    This class has specializations which can be used for dealing with different
    kinds of data like 'images', 'sounds', etc.
    """
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
            
    def initFromCache(self, dataHandlerCache):
        """
        initFromCache(dataHandlerCache : path to cache)
        initialize the dataHandler with a previously created cache 
        
        this helps you use a split (into validation/train/test sets) of your 
        data created previously, so that you can run you experiments, for
        example, measuring accuracy on test set, on the same set of data files
        """
        print('Initializing DataHandler from a previous cache : ', dataHandlerCache)
        
        # read the cache from disk
        assert(os.path.isfile(dataHandlerCache))
        prevCache = np.load(dataHandlerCache).item()
        assert(type(prevCache) == dict)
        
        # set the cache member variable 
        self.m_dataHandlerCache = prevCache
        
        # initialize other member variables based on the previous cache
        self.m_dataDir = prevCache['dataDir']
        self.m_labelsDir = prevCache['labelsDir']
        self.m_dataExtension = prevCache['dataExtension']
        self.m_labelsExtension = prevCache['labelsExtension']
        self.m_trainFraction = prevCache['trainFraction']
        self.m_validationFraction = prevCache['validationFraction']
        self.m_testFraction = prevCache['testFraction']
        
    def isEmpty(self):
        """
        isEmpty() : returns true if the dataHandler is empty
        check(whether empty) is performed on the dataHandlerCache
        """
        if len(self.m_dataHandlerCache) <= 0:
            return True
        else:
            return False
        
    def setDataExtension(self, dataExt):
        """
        setDataExtension(dataExt) : like 'png', 'jpeg', etc.
        specify the extension of your data files
        """
        if dataExt.startswith('.'):
            self.m_dataExtension = dataExt[1:]
        else:
            self.m_dataExtension = dataExt
        
    def getDataExtension(self):
        """
        getDataExtension() : returns extension of the data that the datahandler
        is handling
        """
        return self.m_dataExtension
    
    def setLabelsExtension(self, labelsExt):
        """
        setLabelsExtension(labelsExt) : specify extension that the label files 
        """
        if labelsExt.startswith('.'):
            self.m_labelsExtension = labelsExt[1:]
        else:
            self.m_labelsExtension = labelsExt
        
    def getLabelsExtension(self):
        """
        getLabelsExtension() : get extension of the label files handled by this 
        data handler
        """
        return self.m_labelsExtension
    
    def setDataDir(self, dataDir):
        """
        setDataDir(dataDir) : specify the path of the directory that contains
        the data (same directory should contain both training and testing data)
        """
        assert(os.path.isdir(dataDir))
        self.m_dataDir = os.path.join(dataDir, "") #to add '/' or '\' 
        
    def getDataDir(self):
        """
        getDataDir() : returns the data directory of this dataHandler
        """
        return self.m_dataDir
    
    def setLabelsDir(self, labelsDir):
        """
        setLabelsDir(labelsDir) : specify the directory that contains labels
        same directory must contain labels for both training and testing data
        """
        assert(os.path.isdir(labelsDir))
        self.m_labelsDir = os.path.join(labelsDir, "")
        
    def getLabelsDir(self):
        """
        getLabelsDir() : returns the directory that contains label files of the
        dataHandler
        """
        return self.m_labelsDir
       
    def getDataHandlerCache(self):
        """
        getDataHanderCache() : returns the internal object that contains the 
        partition that the dataHandler has created of all your data into training, 
        test, and validation
        """
        return self.m_dataHandlerCache

    def setTrainFraction(self, trainFraction):
        """
        setTrainFraction(trainFraction) : float 0 < trFrac < 1, specify the 
        fraction of all data that is to be used as training data
        """
        assert(trainFraction >= 0.0 and trainFraction <= 1.0)
        self.m_trainFraction = trainFraction
        self.m_testFraction = 1.0 - trainFraction
        assert(self.m_trainFraction + self.m_testFraction == 1.0)
        
    def getTrainFraction(self):
        """
        getTrainFraction() : get the fraction of data that has been marked as 
        training data in this dataHandler
        """
        return self.m_trainFraction
        
    def setValidationFraction(self, validationFraction):
        """
        setValidationFraction(validationFraction) : float 0 < valFrac < 1, specify the 
        fraction of all data that is to be used as validation data
        """
        assert(validationFraction >= 0.0 and validationFraction <= 1.0)
        self.m_validationFraction = validationFraction
        
    def getValidationFraction(self):
        """
        getValidationFraction() : get the fraction of data that has been marked as 
        validation data in this dataHandler
        """        
        return self.m_validationFraction
    
    def setTestFraction(self, testFraction):
        """
        setTestFraction(validationFraction) : float 0 < testFrac < 1, specify the 
        fraction of all data that is to be used as test data
        """
        assert(testFraction >= 0.0 and testFraction <= 1.0)
        self.m_testFraction =  testFraction
        self.m_trainFraction = 1.0 - testFraction
        assert(self.m_trainFraction + self.m_testFraction == 1.0)
        
    def getTestFraction(self):
        """
        getTestFraction() : get the fraction of data that has been marked as 
        test data in this dataHandler
        """   
        return self.m_testFraction
    
    # helper function
    def getFileNameWoExt(self, fn):
        """
        getFileNameWoExt(fn) : internal funciton to strip a file name of its
        extensions, ex : fn = dumfile.tar.gz.zip produces dumfile
        """
        if os.path.splitext(fn)[1] == '':
            return os.path.splitext(fn)[0]
        else:
            return self.getFileNameWoExt(os.path.splitext(fn)[0])

    def getValidationDataSize(self):
        """
        getValidationDataSize() : get the number of data files marked as 
        validation data by the dataHandler
        """
        assert(len(self.m_dataHandlerCache) > 0)
        return (len(self.m_dataHandlerCache['validation']['data']))
    
    def getTrainDataSize(self):
        """
        getTrainDataSize() : get the number of data files marked as 
        train data by the dataHandler
        """
        assert(len(self.m_dataHandlerCache) > 0)
        return (len(self.m_dataHandlerCache['train']['data']))

    def getTestDataSize(self):
        """
        getTestDataSize() : get the number of data files marked as 
        test data by the dataHandler
        """
        assert(len(self.m_dataHandlerCache) > 0)
        return (len(self.m_dataHandlerCache['test']['data']))

    def buildDataHandlerCache(self):
       """
       buildDataHandlerCache() : splits the data found in the data folder into 
       validation, train and test sets, files are shuffled randomly before 
       performing the split
       """
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
       
       # member variables:
       self.m_dataHandlerCache['dataDir'] = self.m_dataDir
       self.m_dataHandlerCache['labelsDir'] = self.m_labelsDir
       self.m_dataHandlerCache['dataExtension'] = self.m_dataExtension
       self.m_dataHandlerCache['labelsExtension'] = self.m_labelsExtension
       self.m_dataHandlerCache['trainFraction'] = self.m_trainFraction
       self.m_dataHandlerCache['validationFraction'] = self.m_validationFraction
       self.m_dataHandlerCache['testFraction'] = self.m_testFraction
       
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
    """
    Specialization of the data handler to deal with image data
    The additional method here is 'loadData' which allows loading image data
    using OpenCV
    """
    def __init__(self):
        DataHandler.__init__(self);
        
    def loadData(self, imagePaths, grayScale=False):
       """
       loadData(imagePaths, grayScale=False) : grayScale=True if images are to 
       be loaded as grayscale
       imagePaths is a list of strings specifying paths to images to be loaded
       """
       assert(type(imagePaths) == list or type(imagePaths) == str)
       
       if type(imagePaths) == list:
           sampleIm = cv2.imread(imagePaths[0])
           w = sampleIm.shape[1]
           h = sampleIm.shape[0]
           if not grayScale:
               ch = sampleIm.shape[2]
           d = len(imagePaths)
           if not grayScale:
               images = np.zeros((d, h, w, ch))
           else:
               images = np.zeros((d, h, w, 1))
               
           nIm = 0
           for imagePath in imagePaths:
               if not grayScale:
                   image = cv2.imread(imagePath)
                   images[nIm,:,:,:] = image
               else:
                   image = cv2.imread(imagePath)
                   images[nIm,:,:,:] = image[:,:,0].reshape((h, w, 1))
               nIm += 1
           return images
       elif type(imagePaths) == str:
           if not grayScale:
               return cv2.imread(imagePath)
           else:
               return cv2.imread(imagePath).reshape((h, w, 1))
   
#--------------------------------
# specialization of the ImageDataHandler class for handling SFT problem, when
# labels are 3D coordinates of vertices of the template mesh
# this class is designed specifically for a mesh with particular vertices, and 
# UV's
class ImageDataHandler_forSFT(ImageDataHandler):
    """
    Specialization of the ImageDataHandler for data created for SFT 
    """
    def __init__(self):
        ImageDataHandler.__init__(self)
        self.gridVertIdxs = np.load('~gridWithVertexIdxs.npy')
        self.gridWidth = 65
        self.gridHeight = 33
        
    def loadLabels(self, labelPaths):
        """
        loadLabels(labelPaths): load labels for SFT data, saved as .npy files
        with a specific format
        """
        assert(type(labelPaths) == list and len(labelPaths) > 0 and labelPaths[0].endswith('.npy'))
        labels = np.zeros((len(labelPaths), self.gridHeight, self.gridWidth, 3))
        nLabel = 0
        for labelPath in labelPaths:
            rendData = np.load(labelPath).item()
            vertices = rendData['mesh']['vertices'] 
            label = np.zeros((self.gridHeight, self.gridWidth, 3))
            for r in range(self.gridHeight):
                for c in range(self.gridWidth):
                    label[r, c,:] = vertices[int(self.gridVertIdxs[r, c]) ,: ]
            labels[nLabel,:,:,:] = label
            nLabel += 1
        return labels
            
            
        
        