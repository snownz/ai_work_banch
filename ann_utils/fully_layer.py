import tensorflow as tf
import numpy as np
from ann_utils.helper import dropout
from random import random

class FullyLayer(object):

    def __init__(self, size, name, 
                 dropout=0.0, 
                 act=tf.nn.relu, 
                 std=0.02,
                 initializer = None,
                 bias=True):
        
        self.size = size 
        self.initializer = initializer 
        self.std = std 
        self.dropout = dropout 
        self.name = name 
        self.act = act 
        self.bias = bias 

    def __call__(self, x, is_training=False, size=None):

        with tf.variable_scope( self.name ):
            if not size is None:
                self.size = size
            
            if self.initializer is None:
                w = tf.compat.v1.get_variable( "_w_fully", [ x.shape[-1], self.size ], 
                                                initializer = tf.random_normal_initializer( mean = 0.0, stddev = 0.02 ),
                                                trainable = is_training )
            else:
                w = tf.compat.v1.get_variable( "_w_fully", [ x.shape[-1], self.size ], 
                                                initializer = self.initializer,
                                                trainable = is_training )
            
            # setup layer params  
            x = tf.matmul( x, w )

            if self.bias:
                b = tf.compat.v1.get_variable('_b_fully', 
                [ self.size ], initializer = tf.constant_initializer( 0 ), trainable = is_training )
                x += b
            
            if not self.act is None:
                x = self.act( x, name = "{}_act".format(self.name) )

            # setup dropout
            if self.dropout > 0 and is_training:
                x = dropout( x, self.dropout, name = "{}_dp".format(self.name) )
            
            print(x)            
            return x

def mlp( size, name, dp, act, initializer = tf.random_normal_initializer() ):
    return [ FullyLayer( x, '{}_{}'.format( name, i ), act = act, dropout = dp ) for i, x in enumerate( size ) ]
