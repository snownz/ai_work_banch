import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from ann_utils.helper import maxpool2d, avgpool2d, concat
from ann_utils.conv_layer import Conv2DLayer

class IdentityBlock(object):

    def __init__( self,
                  name,
                  k,
                  filters,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu):
            
        l1_f, l2_f, l3_f = filters

        self.c1 = Conv2DLayer( l1_f, 1, 1, '{}_res_a'.format( name ), 
                               dropout, bn, ln, "VALID", act )
        self.c2 = Conv2DLayer( l2_f, k, 1, '{}_res_b'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c3 = Conv2DLayer( l3_f, 1, 1, '{}_res_c'.format( name ),
                               dropout, bn, ln, "VALID", None )
        
        self.act = act                   
                
        
    def __call__(self, x, is_training=False): 

        xi = x
        x = self.c1( x, is_training )
        x = self.c2( x, is_training )
        x = self.c3( x, is_training )
        x = x + xi

        if self.act:
            x = self.act( x )

        print(x)            
        return x

class ConvBlock(object):

    def __init__( self,
                  name,
                  k,
                  filters,
                  s,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu):
            
        l1_f, l2_f, l3_f = filters

        self.c1 = Conv2DLayer( l1_f, 1, s, '{}_res_a'.format( name ), 
                               dropout, bn, ln, "VALID",  act )
        self.c2 = Conv2DLayer( l2_f, k, 1, '{}_res_b'.format( name ),
                               dropout, bn, ln, "SAME",  act )
        self.c3 = Conv2DLayer( l3_f, 1, 1, '{}_res_c'.format( name ),
                               dropout, bn, ln, "VALID",  None )

        self.c4 = Conv2DLayer( l3_f, 1, s, '{}_res_1'.format( name ), 
                               dropout, bn, ln, "VALID",  act )
        
        self.act = act                   
                
        
    def __call__(self, x, is_training=False): 

        xi = self.c4( x, is_training )
        
        x = self.c1( x, is_training )
        x = self.c2( x, is_training )
        x = self.c3( x, is_training )
        x = x + xi

        if self.act:
            x = self.act( x )

        print(x)            
        return x