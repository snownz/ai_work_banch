import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from ann_utils.helper import maxpool2d, avgpool2d, concat, dropout, zero_padding2d, global_average_pool_spartial, bn, ln
from ann_utils.conv_layer import Conv2DLayer

class DesneBlock(object):

    def __init__( self,
                  name,
                  nb_layers,
                  nb_filter,
                  growth_rate,
                  grow_nb_filters = True,
                  dropout = 0.0, bn = False,
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu):
        
        self.layers = []
        for i in range( nb_layers ):
            c = ConvBlock( "{}_{}_block".format( name, i ), growth_rate, dropout = dropout, act = act )
            self.layers.append( c )
            if grow_nb_filters:
                nb_filter += growth_rate
        self.nb_filter = nb_filter
        self.name = name
                        
    def __call__(self, x, is_training=False): 

        concat_feat = x
        for i, l in enumerate( self.layers ):
            x = l( concat_feat, is_training )
            concat_feat = concat( [ concat_feat, x ], axis = 3, name = "{}_{}_concat".format( self.name, i ) )

        print(concat_feat)            
        return concat_feat

class ConvBlock(object):

    def __init__( self,
                  name,
                  filters,
                  dropout = 0.0, bn = False,
                  padding = "SAME",
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu):
        
        self.act = act                   
        self.dp = dropout   
        self.padding = padding              
        self.name = name              
        self.filters = filters            
                        
    def __call__(self, x, is_training=False): 

        inter_channel = x.shape[-1] * 4  

        wc1 = tf.compat.v1.get_variable( "{}_wc1".format( self.name ), 
                               [ 1, 1, x.shape[-1], inter_channel ], 
                               initializer = tf.random_normal_initializer( stddev = 0.02 ), trainable = is_training )

        wc2 = tf.compat.v1.get_variable( "{}_wc2".format( self.name ), 
                               [ 3, 3, inter_channel, self.filters ], 
                               initializer = tf.random_normal_initializer( stddev = 0.02 ), trainable = is_training )     

        # 1x1 Convolution (Bottleneck layer)
        x = bn( x, is_training )
        if self.act:
            x = self.act( x )
        x = tf.nn.conv2d( x, wc1, strides = ( 1, 1, 1, 1 ), padding = self.padding )
        if self.dp > 0:
            x = dropout( x, self.dp )

        # 3x3 Convolution
        x = bn( x, is_training )
        if self.act:
            x = self.act( x )
        x = x = tf.nn.conv2d( x, wc2, strides = ( 1, 1, 1, 1 ), padding = self.padding ) 
        if self.dp > 0:
            x = dropout( x, self.dp )

        print(x)            
        return x

class TransitionBlock(object):

    def __init__( self,
                  name,
                  compression,
                  dropout = 0.0, bn = False,
                  padding = "SAME",
                  l1 = 0.0, l2 = 0.0, 
                  act = tf.nn.leaky_relu):
        
        self.act = act                   
        self.dp = dropout   
        self.padding = padding              
        self.name = name              
        self.compression = compression            

    def __call__(self, x, is_training=False): 

        inter_channel = x.shape.as_list()[-1]

        wc1 = tf.compat.v1.get_variable( "{}_wc1".format( self.name ), 
                               [ 1, 1, x.shape[-1], int( inter_channel * self.compression ) ], 
                               initializer = tf.random_normal_initializer( stddev = 0.02 ), trainable = is_training )

        # 1x1 Convolution (Bottleneck layer)
        x = bn( x, is_training )
        if self.act:
            x = self.act( x )
        x = tf.nn.conv2d( x, wc1, strides = ( 1, 1, 1, 1 ), padding = self.padding )
        if self.dp > 0:
            x = dropout( x, self.dp )

        x = avgpool2d( x, 2, 2 )

        print(x)            
        return x