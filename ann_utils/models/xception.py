import sys
sys.path.append('../../')

import tensorflow as tf
import numpy as np

from ann_utils.helper import maxpool2d, avgpool2d
from ann_utils.conv_layer import Conv2DLayer, SeparableConv2DLayer

'''
  Xception: Deep Learning with Depthwise Separable Convolutions
      Font: https://arxiv.org/pdf/1610.02357.pdf
'''
class Xception(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu ):

        self.act = act

        #===========ENTRY FLOW==============
        # block1
        self.b1c1 = Conv2DLayer( 32, 3, 2, '{}_block1_conv1'.format( name ), 
                                 dropout, bn, ln, "VALID", act ) 
        self.b1c2 = Conv2DLayer( 64, 3, 1, '{}_block1_conv2'.format( name ),
                                 dropout, bn, ln, "VALID", act ) 
        self.b1r1 = Conv2DLayer( 128, 1, 2, '{}_block1_res_conv'.format( name ),
                                 dropout, bn, ln, padding, None ) 
        
        # block2
        self.b2c1 = SeparableConv2DLayer( 128, 3, 1, '{}_block2_dws_conv1'.format( name ),
                                 dropout, bn, ln, padding, act ) 
        self.b2c2 = SeparableConv2DLayer( 128, 3, 1, '{}_block2_dws_conv2'.format( name ),
                                 dropout, bn, ln, padding, None ) 
        self.b2r1 = Conv2DLayer( 256, 1, 2, '{}_block2_res_conv'.format( name ),
                                 dropout, bn, ln, padding, None )
        
        # block3
        self.b3c1 = SeparableConv2DLayer( 256, 3, 1, '{}_block3_dws_conv1'.format( name ),
                                 dropout, bn, ln, padding, act ) 
        self.b3c2 = SeparableConv2DLayer( 256, 3, 1, '{}_block3_dws_conv2'.format( name ),
                                 dropout, bn, ln, padding, None ) 
        self.b3r1 = Conv2DLayer( 728, 1, 2, '{}_block3_res_conv'.format( name ),
                                 dropout, bn, ln, padding, None )

        # block4
        self.b4c1 = SeparableConv2DLayer( 728, 3, 1, '{}_block4_dws_conv1'.format( name ),
                                 dropout, bn, ln, padding, act ) 
        self.b4c2 = SeparableConv2DLayer( 728, 3, 1, '{}_block4_dws_conv2'.format( name ),
                                 dropout, bn, ln, padding, act )

        #===========MIDDLE FLOW===============
        self.block = []
        for i in range(8):
            block_prefix = 'block%s_' % (str(i + 5))
            self.block.append( 
                [
                    SeparableConv2DLayer( 728, 3, 1, '{}_block_{}_dws_conv1'.format( name, block_prefix ),
                                        dropout, bn, ln, padding, act ),
                    SeparableConv2DLayer( 728, 3, 1, '{}_block_{}_dws_conv2'.format( name, block_prefix ),
                                      dropout, bn, ln, padding, act ),
                    SeparableConv2DLayer( 728, 3, 1, '{}_block_{}_dws_conv3'.format( name, block_prefix ),
                                      dropout, bn, ln, padding, None )
                ]
            )
        
        #========EXIT FLOW============
        self.er0 = Conv2DLayer( 1024, 1, 2, '{}_block12_res_conv'.format( name ),
                                dropout, bn, ln, padding, None )

        self.ec0 = SeparableConv2DLayer( 728, 3, 1, '{}_block13_dws_conv1'.format( name ),
                                         dropout, bn, ln, padding, act )
        self.ec1 = SeparableConv2DLayer( 1024, 3, 1, '{}_block13_dws_conv2'.format( name ),
                                         dropout, bn, ln, padding, None )
        
        self.ec2 = SeparableConv2DLayer( 1536, 3, 1, '{}_block14_dws_conv1'.format( name ),
                                         dropout, bn, ln, padding, act )
        self.ec3 = SeparableConv2DLayer( 2048, 3, 1, '{}_block14_dws_conv2'.format( name ),
                                         dropout, bn, ln, padding, act )

        
    def __call__(self, x, is_training=False):

        #===========ENTRY FLOW==============
        # Block 1
        x = self.b1c1( x, is_training )
        x = self.b1c2( x, is_training )
        res = self.b1r1( x, is_training )
        
        # Block 2
        x = self.b2c1( x, is_training )
        x = self.b2c2( x, is_training )
        x = maxpool2d( x, 3, 2 )
        x = tf.add( x, res )
        res = self.b2r1( x, is_training )
        
        # Block 3
        x = self.act( x )
        x = self.b3c1( x, is_training )
        x = self.b3c2( x, is_training )
        x = maxpool2d( x, 3, 2 )
        x = tf.add( x, res )
        res = self.b3r1( x, is_training )
        
        # Block 4
        x = self.act( x )
        x = self.b4c1( x, is_training )
        x = self.b4c2( x, is_training )
        x = maxpool2d( x, 3, 2 )
        x = tf.add( x, res )

        #===========MIDDLE FLOW===============
        for i in range(8):
            res = x
            for n in self.block[i]:
                x = self.act( x )
                x = n( x, is_training )
            x = tf.add( x, res )
        
        #========EXIT FLOW============
        res = self.er0( x, is_training )
        
        x = self.act( x )
        x = self.ec0( x, is_training )
        x = self.ec1( x, is_training )
        x = maxpool2d( x, 3, 2 )
        x = tf.add( x, res )
        
        x = self.ec2( x, is_training )
        x = self.ec3( x, is_training )

        print(x)            
        return x
        