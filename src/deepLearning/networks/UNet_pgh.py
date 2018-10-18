"""
Author : Pamir Ghimire, 15 Oct, 2018
This is a Unet base class which can be inherited from to create a trainable
UNet class or a UNet class that might be used for just making inference
"""
import tensorflow as tf
from abc import ABC, abstractmethod

class UNet(ABC): #inherit from ABC : Abstract Base Class
    """
    This is an abstract class!
    The UNet architecture implemented in this class follows the one described
    in the paper that introduced it : 
        Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    The difference here is the use of batch normalization
    The encoder part contains 5 levels of convolutions (4 max-poolings) and 
    the decoder part contains 4 levels of convolutions (4 transpose convolutions)
    
    The final output is produced by applying a convolution to the output of the 
    last convolutional level in the decoder
    
    # note : econvx_y and dconvx_y correspond to convolutional filters in 
    # encoder and decoder parts of the Unet (dconv does not imply
    # deconvolutional layer related information)
    
    Classes inheriting form UNet are free to extend the architecture to produce
    an output of arbitrary dimensions
    
    Input Size : batch x H x W x 3
        H = 572
        W = 572

    How to use: (sample code)
    # Test Code : How to use UNetToTrain/UNetToPredict
    import numpy as np
    
    myUnet = UNetToTrain()
    myUnet.setInitFromScratch(True)
    myUnet.initializeWeights()
    
    dumImages = np.random.rand(100, 572, 572, 3)
    with tf.Session() as sess:
        feed_dict = {myUnet.m_inputStack:dumImages}
        sess.run(myUnet.m_inputStack, feed_dict)    


    """
    def __init__(self):
        tf.reset_default_graph() # free up tensorflow's cache
        
        self.m_inputImageWidth = 480;
        self.m_inputImageHeight = 480;
        self.m_inputChannels = 3
        self.m_batchNormalization = True
        self.m_inputStack = None
        self.m_unet = None
        
        # network architecture related variables, filter shapes are (h, w, inChannels, outChannels)
        self.m_trainableWeights = {'econv1_1':(3, 3, 3, 64), 'econv1_1b': (64,), \
                                'econv1_2':(3, 3, 64, 64), 'econv1_2b': (64,), \
                               'econv2_1':(3, 3, 64, 128), 'econv2_1b': (128,), \
                               'econv2_2':(3, 3, 128, 128), 'econv2_2b': (128,), \
                               'econv3_1':(3, 3, 128, 256), 'econv3_1b': (256,), \
                               'econv3_2':(3, 3, 256, 256), 'econv3_2b': (256,), \
                               'econv4_1':(3, 3, 256, 512), 'econv4_1b': (512,), \
                               'econv4_2':(3, 3, 512, 512), 'econv4_2b': (512,), \
                               'econv5_1':(3, 3, 512, 1024), 'econv5_1b': (1024,), \
                               'econv5_2':(3, 3, 1024, 1024), 'econv5_2b': (1024,), 
                               'upConv4':(2, 2, 512, 1024),\
                               'dconv4_1':(3, 3, 1024, 512), 'dconv4_1b': (512,), \
                               'dconv4_2':(3, 3, 512, 512), 'dconv4_2b': (512,), \
                               'upConv3':(2, 2, 256, 512), \
                               'dconv3_1':(3, 3, 512, 256), 'dconv3_1b': (256,), \
                               'dconv3_2':(3, 3, 256, 256), 'dconv3_2b': (256,), \
                               'upConv2':(2, 2, 128, 256), \
                               'dconv2_1':(3, 3, 256, 128), 'dconv2_1b': (128,), \
                               'dconv2_2':(3, 3, 128, 128), 'dconv2_2b': (128,), \
                               'upConv1':(2, 2, 64, 128), \
                               'dconv1_1':(3, 3, 128, 64), 'dconv1_1b': (64,), \
                               'dconv1_2':(3, 3, 64, 64), 'dconv1_2b': (64,)}
        
    def initializeWeights(self):
        self.m_inputStack = tf.placeholder(tf.float32, (None, self.m_inputImageHeight, self.m_inputImageWidth, self.m_inputChannels))
        self.m_unet = self.UnetArch(self.m_inputStack)

    def UnetArch(self, x):
        """
        UnetArch(x): 'x' is input tensor of dimensions (N, H, W, C)
        """
        print('shape of x : ', x.get_shape())
        trainable = True
        # Encoder layers:
        econv1_1 = self.conv_layer(x, "econv1_1", trainable=trainable)
        econv1_2 = self.conv_layer(econv1_1, "econv1_2", trainable=trainable)
        print('shape of encov1_2 : ', econv1_2.get_shape())
        pool1 = self.max_pool(econv1_2, "pool1")
        print('shape of pool1 : ', pool1.get_shape())
        
        econv2_1 = self.conv_layer(pool1, "econv2_1", trainable=trainable)
        econv2_2 = self.conv_layer(econv2_1, "econv2_2", trainable=trainable)
        print('shape of encov2_2 : ', econv2_2.get_shape())
        pool2 = self.max_pool(econv2_2, "pool2")
        print('shape of pool2 : ', pool2.get_shape())
        
        econv3_1 = self.conv_layer(pool2, "econv3_1", trainable=trainable)
        econv3_2 = self.conv_layer(econv3_1, "econv3_2", trainable=trainable)
        print('shape of encov3_2 : ', econv3_2.get_shape())
        pool3 = self.max_pool(econv3_2, "pool3")
        print('shape of pool3 : ', pool3.get_shape())

        econv4_1 = self.conv_layer(pool3, "econv4_1", trainable=trainable)
        print('shape of encov4_1 : ', econv4_1.get_shape())
        econv4_2 = self.conv_layer(econv4_1, "econv4_2", trainable=trainable)
        pool4 = self.max_pool(econv4_2, "pool4")
        print('shape of pool4 : ', pool4.get_shape())

        econv5_1 = self.conv_layer(pool4, "econv5_1", trainable=trainable)
        econv5_2 = self.conv_layer(econv5_1, "econv5_2", trainable=trainable)
        print('shape of encov5_2 : ', econv5_2.get_shape())

        # Decoder layers
        upConv4 = self.upConv_layer(econv5_2, "upConv4", trainable=trainable)
        concat4 = self.concat_layer(econv4_2, upConv4, "concat4")
        dconv4_1 = self.conv_layer(concat4, "dconv4_1", trainable=trainable)
        dconv4_2 = self.conv_layer(dconv4_1, "dconv4_2", trainable=trainable)
        
        upConv3 = self.upConv_layer(dconv4_2, "upConv3", trainable=trainable)
        concat3 = self.concat_layer(econv3_2, upConv3, "concat3")
        dconv3_1 = self.conv_layer(concat3, "dconv3_1", trainable=trainable)
        dconv3_2 = self.conv_layer(dconv3_1, "dconv3_2", trainable=trainable)
        
        upConv2 = self.upConv_layer(dconv3_2, "upConv2", trainable=trainable)
        concat2 = self.concat_layer(econv2_2, upConv2, "concat2")
        dconv2_1 = self.conv_layer(concat2, "dconv2_1", trainable=trainable)
        dconv2_2 = self.conv_layer(dconv2_1, "dconv2_2", trainable=trainable)
        
        upConv1 = self.upConv_layer(dconv2_2, "upConv1", trainable=trainable)
        concat1 = self.concat_layer(econv1_2, upConv1, "concat1")
        dconv1_1 = self.conv_layer(concat1, "dconv1_1", trainable=trainable)
        dconv1_2 = self.conv_layer(dconv1_1, "dconv1_2", trainable=trainable)
        
        return dconv1_2
        
    def getInputImageWidth(self):
        return self.m_inputImageWidth
    
    def setInputImageWidth(self, imwidth):
        assert(imwidth > 0)
        self.m_inputImageWidth = imwidth
        
    def getInputImageHeight(self):
        return self.m_inputImageHeight
        
    def setInputImageHeight(self, imheight):
        assert(imheight > 0)
        return self.m_inputImageHeight

    def conv_layer(self, bottom, name, trainable=True):
        filt = self.get_conv_filter(name, trainable)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = self.get_bias(name+'b', trainable)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias, name=name)
        return relu
    
    def upConv_layer(self, bottom, name, trainable=True):
        """
        upConv_layer(bottom, name, trainable=True):
        reference : dynamically specify batch size to compute output_shape
        https://stackoverflow.com/questions/46885191/tf-nn-conv2d-transpose-output-shape-dynamic-batch-size
        """
        filt = self.get_conv_filter(name, trainable)
        
        batch = tf.shape(bottom)[0] 
        height = int(2*bottom.get_shape()[1])
        width = int(2*bottom.get_shape()[2])
        channels = int(int(bottom.get_shape()[3])/2.0)
        #output_shape = tf.Variable([batch, height, width, channels], \
                                   #tf.int32, name=name+'_outputShape', trainable=False)
        output_shape = [batch, height, width, channels]
        conv = tf.nn.conv2d_transpose(bottom, filt, output_shape, strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
        
        return conv
        
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
  
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def concat_layer(self, fromEncoder, upSampled, name):
        """
        concat_layer(self, upsampled, fromEncoder): concatenate two tensors
        """
        return tf.concat([fromEncoder, upSampled], axis=-1, name=name)

    @abstractmethod
    def get_conv_filter(self, name, trainable):
        pass
    
    @abstractmethod
    def get_bias(self, name, trainable):
        pass

    # save model
    # load previously saved model
    
#---------------------------------------------

class UNetToTrain(UNet):
    """
     Specialization of the u-net class for training a model by initializing weights
     from scratch (or from a weights file (ckpt or .npy))
     
     This specialization does not define a mapping from the final feature layer
     to the desired output map (like the 1x1 convolutions in original UNet paper)
     
     This prevents specific cost functions from being defined in the class
     
     To train the UNet to produce a specific kind of output, check the
     specializations below
    """
    def __init__(self):
        UNet.__init__(self)
        self.m_initFromScratch = True
        
    def setInitFromScratch(self, initFromScratch):
        self.m_initFromScratch = initFromScratch
        
    def getInitFromScratch(self):
        return self.m_initFromScratch
    
    def get_conv_filter(self, name, trainable):
        """
        returns a convolutional filter initialized from sratch or from
        a weights file
        """
        if self.m_initFromScratch:
            return tf.get_variable(name=name, \
                                   shape=self.m_trainableWeights[name], \
                                   initializer = tf.contrib.layers.xavier_initializer(), \
                                   trainable=trainable)
        else:
            #@todo : add support to init weights from a file (ckpt or .npy)
            pass
        
    def get_bias(self, name, trainable):
        if self.m_initFromScratch:
            return tf.get_variable(name=name,\
                                     shape=self.m_trainableWeights[name],\
                                     initializer = tf.contrib.layers.xavier_initializer(), \
                                     trainable=trainable)
        else:
            #@todo : add support to init weights from a file (ckpt or .npy)
            pass

        
#---------------------------------------------
class UNetToTrainForSFT(UNetToTrain):
    """
     Specialization of the trainable UNet class to predictvertices of a template
     You can find the specifications of the input in UNet class
     
     In this specialization, to the output of the UNet, we append 2 conv. 
     layers, the vertex coordinates are predicted by the second layer which is 
     the output of a 1x1 convolutional layer
     The 3 convolutional layers apply padding such that the width and height of
     final feature map produced by the UNet do not change
     Because the number of vertex locations we want to predict are less than 
     the number of 'pixels' in the output of the 1x1 conv. layer, the network
     is trained to produce the vertex locations only in the upper left corner
     of the output
     
     To train this class, you have to provide labels in form of a tensor 
     that describe (in a grid) the 3D coordinates of vertices of a template 
     mesh corresponding to the object in the input image
     
     
     @todo : add 3 convolutional layers after the final feature map (two 3x3 convs,
     1 1x1 conv, predictions to be in a 'crop' of the output of the 1x1 conv layer)
    """
    def __init__(self):
        UNetToTrain.__init__(self)
        self.m_vertexPredictions = None
        self.m_label = None
        
        # weights specific to this specialization
        self.m_trainableWeights['dconv1_3'] = (3, 3, 64, 3)
        self.m_trainableWeights['dconv1_3b'] = (3,)
        self.m_trainableWeights['dconv1_4'] = (1, 1, 3, 3)
        self.m_trainableWeights['dconv1_4b'] = (3,)

    def initializeWeights(self):
        super(UNetToTrain, self).initializeWeights()
        #self.m_vertexPredictions = self.getVertexPredictions(self.m_unet)
        

#---------------------------------------------
#How to use: (sample code)
    # Test Code : How to use UNetToTrain/UNetToPredict
import numpy as np

myUnet = UNetToTrain()
myUnet.setInitFromScratch(True)

dumImages = np.random.rand(10, 480, 480, 3)

with tf.Graph().as_default():
    myUnet.initializeWeights()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
     
        feed_dict = {myUnet.m_inputStack:dumImages}
        sess.run(myUnet.m_unet, feed_dict=feed_dict)    
    