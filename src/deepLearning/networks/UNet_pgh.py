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
    """
    def __init__(self):
        tf.reset_default_graph() # free up tensorflow's cache
        
        self.m_inputImageWidth = 572;
        self.m_inputImageHeight = 572;
        self.m_inputChannels = 3
        self.m_batchNormalization = True
        self.m_inputStack = tf.placeholder(tf.float16, (None, self.m_inputImageHeight, self.m_inputImageWidth, self.m_inputChannels))
       
        
        # network architecture related variables
        self.m_variableNames = {'econv1_1':(3, 3, 64), 'econv1_1b': (64,), \
                                'econv1_2':(3, 3, 64), 'econv1_2b': (64,), \
                               'econv2_1':(3, 3, 128), 'econv2_1b': (128,), \
                               'econv2_2':(3, 3, 128), 'econv2_2b': (128,), \
                               'econv3_1':(3, 3, 256), 'econv3_1b': (256,), \
                               'econv3_2':(3, 3, 256), 'econv3_2b': (256,), \
                               'econv4_1':(3, 3, 512), 'econv4_1b': (512,), \
                               'econv4_2':(3, 3, 512), 'econv4_2b': (512,), \
                               'econv5_1':(3, 3, 1024), 'econv5_1b': (1024,), \
                               'econv5_2': (), 'econv5_2b', \
                               'dconv4_1', 'dconv4_1b', 'dconv4_2', 'dconv4_2b',\
                               'dconv3_1', 'dconv3_1b', 'dconv3_2', 'dconv3_2b',\
                               'dconv2_1', 'dconv2_1b', 'dconv2_2', 'dconv2_2b',\
                               'dconv1_1', 'dconv1_1b', 'dconv1_2', 'dconv1_2b',\
                               'donv1_3', 'dconv1_3b'] 
        
        
        # abstract method for initialization:
        #       trainer should initialize them from scratch 
        #       tester/predictor should initialize them from a file
    
    def UnetArch(self, x):
        trainable = True
        # Encoder layers:
        econv1_1 = self.conv_layer(x, "econv1_1", trainable=trainable)
        econv1_2 = self.conv_layer(econv1_1, "econv1_2", trainable=trainable)    
        pool1 = self.max_pool(econv1_2, "pool1")
        
        econv2_1 = self.conv_layer(pool1, "econv2_1", trainable=trainable)
        econv2_2 = self.conv_layer(econv2_1, "econv2_2", trainable=trainable)
        pool2 = self.max_pool(econv2_2, "pool2")

        econv3_1 = self.conv_layer(pool2, "econv3_1", trainable=trainable)
        econv3_2 = self.conv_layer(econv3_1, "econv3_2", trainable=trainable)
        pool3 = self.max_pool(econv3_2, "pool3")

        econv4_1 = self.conv_layer(pool3, "econv4_1", trainable=trainable)
        econv4_2 = self.conv_layer(econv4_1, "econv4_2", trainable=trainable)
        pool4 = self.max_pool(econv4_2, "pool4")

        econv5_1 = self.conv_layer(pool4, "econv5_1", trainable=trainable)
        econv5_2 = self.conv_layer(econv5_1, "econv5_2", trainable=trainable)

        # Decoder layers
        transConv4 = self.deconv_layer(econv5_2, "transConv4", trainable=trainable)
        concat4 = self.concat_layer(econv4_2, transConv4, "concat4")
        dconv4_1 = self.conv_layer(concat4, "dconv4_1", trainable=trainable)
        dconv4_2 = self.conv_layer(dconv4_1, "dconv4_2", trainable=trainable)
        
        transConv3 = self.deconv_layer(dconv4_2, "transConv3", trainable=trainable)
        concat3 = self.concat_layer(econv3_2, transConv3, "concat3")
        dconv3_1 = self.conv_layer(concat3, "dconv3_1", trainable=trainable)
        dconv3_2 = self.conv_layer(dconv3_1, "dconv3_2", trainable=trainable)

        transConv2 = self.deconv_layer(dconv3_2, "transConv2", trainable=trainable)
        concat2 = self.concat_layer(econv2_2, transConv2, "concat2")
        dconv2_1 = self.conv_layer(concat2, "dconv2_1", trainable=trainable)
        dconv2_2 = self.conv_layer(dconv2_1, "dconv2_2", trainable=trainable)

        transConv1 = self.deconv_layer(dconv2_2, "transConv1", trainable=trainable)
        concat1 = self.concat_layer(econv1_2, transConv1, "concat1")
        dconv1_1 = self.conv_layer(concat1, "dconv1_1", trainable=trainable)
        dconv1_2 = self.conv_layer(dconv1_1, "dconv1_2", trainable=trainable)

        # output layer
        out = self.conv_layer(dconv1_2, "dconv1_3", trainable=trainable)
        
        return out
        
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

        conv_biases = self.get_bias(name, trainable)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias, name=name)
        return relu
    
    def deconv_layer(self, bottom, name, trainable=True):
        filt = self.get_conv_filter(name, trainable)
#        output_shape = (tf.shape(self.m_inputStack)[0], )
        conv = tf.nn.conv2d_transpose(bottom, filt, output_shape, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
  
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    @abstractmethod
    def get_conv_filter(self, name, trainable):
        pass
    
    @abstractmethod
    def get_bias(self, name, trainable):
        pass

    # save weights method
    # load weights method
    
#---------------------------------------------
# specialization of the u-net class for training a model from scratch (or by
# starting with weights from a pre-training)
class UNetToTrain(UNet):
    def __init__(self):
        UNet.__init__(self)
        self.m_initFromScratch = True
        
    def setInitFromScratch(self, initFromScratch):
        self.m_initFromScratch = initFromScratch
        
    def getInitFromScratch(self):
        return self.m_initFromScratch
    
    def get_conv_filter(self, name, trainable):
        if self.m_initFromScratch:
            return 