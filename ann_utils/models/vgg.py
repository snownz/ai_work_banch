import sys
sys.path.append('../')

import tensorflow as tf

from ann_utils.conv_layer import Conv2DLayer
from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import avgpool2d, zero_padding2d, maxpool2d

class VGG16(object):

    def __init__( self,
                  name, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        # stage 1
        self.st1_1 = Conv2DLayer( 64, 3, 1, '{}_st1_1'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st1_2 = Conv2DLayer( 64, 3, 1, '{}_st1_2'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        # stage 2
        self.st2_1 = Conv2DLayer( 128, 3, 1, '{}_st2_1'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st2_2 = Conv2DLayer( 128, 3, 1, '{}_st2_2'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        # stage 3
        self.st3_1 = Conv2DLayer( 256, 3, 1, '{}_st3_1'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st3_2 = Conv2DLayer( 256, 3, 1, '{}_st3_2'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st3_3 = Conv2DLayer( 256, 3, 1, '{}_st3_3'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        # stage 4
        self.st4_1 = Conv2DLayer( 512, 3, 1, '{}_st4_1'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st4_2 = Conv2DLayer( 512, 3, 1, '{}_st4_2'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st4_3 = Conv2DLayer( 512, 3, 1, '{}_st4_3'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        # stage 5
        self.st5_1 = Conv2DLayer( 512, 3, 1, '{}_st5_1'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st5_2 = Conv2DLayer( 512, 3, 1, '{}_st5_2'.format( name ), 
                               0.0, bn, ln, "SAME", act )

        self.st5_3 = Conv2DLayer( 512, 3, 1, '{}_st5_3'.format( name ), 
                               0.0, bn, ln, "SAME", act )
        

    def __call__(self, x, is_training=False):

        x = self.st1_1( x, is_training )
        x = self.st1_2( x, is_training )
        x = maxpool2d( x, 2, 2 )

        x = self.st2_1( x, is_training )
        x = self.st2_2( x, is_training )
        x = maxpool2d( x, 2, 2 )

        x = self.st3_1( x, is_training )
        x = self.st3_2( x, is_training )
        x = self.st3_3( x, is_training )
        x = maxpool2d( x, 2, 2 )

        x = self.st4_1( x, is_training )
        x = self.st4_2( x, is_training )
        x = self.st4_3( x, is_training )
        x = maxpool2d( x, 2, 2 )

        x = self.st5_1( x, is_training )
        x = self.st5_2( x, is_training )
        x = self.st5_3( x, is_training )
        x = maxpool2d( x, 2, 2 )

        return x
