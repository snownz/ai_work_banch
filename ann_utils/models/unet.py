import sys
sys.path.append('../../')

import tensorflow as tf
import numpy as np

from ann_utils.helper import maxpool2d, avgpool2d
from ann_utils.conv_layer import Conv2DLayer, Deconv2DLayer

class Unet(object):

    def __init__( self,
                  name, scope,
                  chanels,
                  dropout = 0.0, bn = False,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu ):

        self.act = act

        # downsample
        self.c0_0 = Conv2DLayer( 64, 3, 1,  "{}_c00".format( name ), bn = bn, ln = ln, act = act )

        self.c1_0 = Conv2DLayer( 64,  3, 1, "{}_c10".format( name ), bn = bn, ln = ln, act = act )
        self.c1_1 = Conv2DLayer( 128, 3, 1, "{}_c11".format( name ), bn = bn, ln = ln, act = act )

        self.c2_0 = Conv2DLayer( 128, 3, 1, "{}_c20".format( name ), bn = bn, ln = ln, act = act )
        self.c2_1 = Conv2DLayer( 256, 3, 1, "{}_c21".format( name ), bn = bn, ln = ln, act = act )
        self.c2_2 = Conv2DLayer( 256, 3, 1, "{}_c22".format( name ), bn = bn, ln = ln, act = act )

        self.c3_0 = Conv2DLayer( 256, 3, 1, "{}_c30".format( name ), bn = bn, ln = ln, act = act )
        self.c3_1 = Conv2DLayer( 512, 3, 1, "{}_c31".format( name ), bn = bn, ln = ln, act = act )
        self.c3_2 = Conv2DLayer( 512, 3, 1, "{}_c32".format( name ), bn = bn, ln = ln, act = act )

        self.c4_0 = Conv2DLayer( 512, 3, 1, "{}_c40".format( name ), bn = bn, ln = ln, act = act )
        self.c4_1 = Conv2DLayer( 512, 3, 1, "{}_c41".format( name ), bn = bn, ln = ln, act = act )
        self.c4_2 = Conv2DLayer( 512, 3, 1, "{}_c42".format( name ), bn = bn, ln = ln, act = act )

        self.c5_0 = Conv2DLayer( 512, 3, 1, "{}_c50".format( name ), bn = bn, ln = ln, act = act )
        self.c5_1 = Conv2DLayer( 512, 3, 1, "{}_c51".format( name ), bn = bn, ln = ln, act = act )
        
        # upsample
        self.d0_0 = Deconv2DLayer( 256, 2, 2, "{}_d00".format( name ), bn = bn, ln = ln, act = act )
        self.d0_1 = Deconv2DLayer( 512, 3, 1, "{}_d01".format( name ), bn = bn, ln = ln, act = act )

        self.d1_0 = Deconv2DLayer( 256, 2, 2, "{}_d10".format( name ), bn = bn, ln = ln, act = act )
        self.d1_1 = Deconv2DLayer( 512, 3, 1, "{}_d11".format( name ), bn = bn, ln = ln, act = act )

        self.d2_0 = Deconv2DLayer( 128, 2, 2, "{}_d20".format( name ), bn = bn, ln = ln, act = act )
        self.d2_1 = Deconv2DLayer( 256, 3, 1, "{}_d21".format( name ), bn = bn, ln = ln, act = act )

        self.d3_0 = Deconv2DLayer( 64,  2, 2, "{}_d30".format( name ), bn = bn, ln = ln, act = act )
        self.d3_1 = Deconv2DLayer( 128, 3, 1, "{}_d31".format( name ), bn = bn, ln = ln, act = act )

        self.d4_0 = Deconv2DLayer( 32, 2, 2,  "{}_d40".format( name ), bn = bn, ln = ln, act = act )
        self.d4_1 = Deconv2DLayer( chanels, 3,  1,  "{}_d41".format( name ), bn = False, act = tf.nn.sigmoid )
       
        
    def __call__(self, x, is_training=False):
        
        x0 = self.c0_0( x, is_training )
        x  = maxpool2d( x0, 2, 2 )

        x  = self.c1_0( x, is_training )
        x1 = self.c1_1( x, is_training )
        x  = maxpool2d( x1, 2, 2 )

        x  = self.c2_0( x, is_training )
        x2 = self.c2_1( x, is_training )
        x  = maxpool2d( x2, 2, 2 )

        x  = self.c3_0( x, is_training )
        x3 = self.c3_1( x, is_training )
        x  = maxpool2d( x3, 2, 2 )

        x  = self.c4_0( x, is_training )
        x4 = self.c4_1( x, is_training )
        x  = maxpool2d( x4, 2, 2 )

        x = self.c5_0( x, is_training )
        x = self.c5_1( x, is_training )

        x = self.d0_0( x, is_training )
        x = tf.concat( [ x, x4 ], axis = 3 )
        x = self.d0_1( x, is_training )

        x = self.d1_0( x, is_training )
        x = tf.concat( [ x, x3 ], axis = 3 )
        x = self.d1_1( x, is_training )

        x = self.d2_0( x, is_training )
        x = tf.concat( [ x, x2 ], axis = 3 )
        x = self.d2_1( x, is_training )

        x = self.d3_0( x, is_training )
        x = tf.concat( [ x, x1 ], axis = 3 )
        x = self.d3_1( x, is_training )

        x = self.d4_0( x, is_training )
        x = tf.concat( [ x, x0 ], axis = 3 )
        x = self.d4_1( x, is_training )
                
        print(x)            
        return x
        