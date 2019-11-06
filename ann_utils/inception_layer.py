import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from ann_utils.helper import maxpool2d, avgpool2d, concat
from ann_utils.conv_layer import Conv2DLayer
from ann_utils.models.squeeze_n_excitation import SqueezeNExcitation

class InceptionV2StemLayer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu):

        self.name = name
                
        self.c1 = Conv2DLayer( 32, 3, 2, '{}_incep_resnet_stem_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )
        self.c2 = Conv2DLayer( 32, 3, 1, '{}_incep_resnet_stem_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c3 = Conv2DLayer( 64, 3, 1, '{}_incep_resnet_stem_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c4 = Conv2DLayer( 96, 3, 2, '{}_incep_resnet_stem_4'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c5_0 = Conv2DLayer( 64, 1, 1, '{}_incep_resnet_stem_5_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c5_1 = Conv2DLayer( 64, (1,7), 1, '{}_incep_resnet_stem_5_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c5_2 = Conv2DLayer( 64, (7,1), 1, '{}_incep_resnet_stem_5_2'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c5_3 = Conv2DLayer( 96, 3, 1, '{}_incep_resnet_stem_5_3'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c6_0 = Conv2DLayer( 64, 1, 1, '{}_incep_resnet_stem_6_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c6_1 = Conv2DLayer( 96, 3, 1, '{}_incep_resnet_stem_6_1'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c7 = Conv2DLayer( 192, 3, 2, '{}_incep_resnet_stem_7'.format( name ),
                               dropout, bn, ln, "SAME", act )                           
                
        
    def __call__(self, x, is_training=False): 

        x = self.c1( x, is_training )
        x = self.c2( x, is_training )
        x = self.c3( x, is_training )

        x1 = self.c4( x, is_training )
        x2 = maxpool2d( x, 2, 2, padding="VALID" )

        x = concat( [ x1, x2 ], axis = 3, name = "{}_concat1".format( self.name )  )

        x1 = self.c5_0( x, is_training )
        x2 = self.c6_0( x, is_training )

        x1 = self.c5_1( x1, is_training )
        x1 = self.c5_2( x1, is_training )
        x1 = self.c5_3( x1, is_training )

        x2 = self.c6_1( x2, is_training )

        x = concat( [ x1, x2 ], axis = 3, name = "{}_concat2".format( self.name )  )

        x1 = self.c7( x, is_training )
        x2 = maxpool2d( x, 2, 2, padding="VALID" )

        x = concat( [ x1, x2 ], axis = 3, name = "{}_concat3".format( self.name )  )
        
        print(x)            
        return x

class InceptionResnetAV2Layer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.act = act
        self.name = name
                
        self.c0_0 = Conv2DLayer( 32, 1, 1, '{}_incep_resnet_a_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 32, 1, 1, '{}_incep_resnet_a_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 32, 3, 1, '{}_incep_resnet_a_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 32, 1, 1, '{}_incep_resnet_a_2_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c2_1 = Conv2DLayer( 48, 3, 1, '{}_incep_resnet_a_2_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c2_2 = Conv2DLayer( 64, 3, 1, '{}_incep_resnet_a_2_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c3_0 = Conv2DLayer( 384, 1, 1, '{}_incep_resnet_a_3_0'.format( name ),
                               dropout, bn, ln, "SAME", None )

                                   
    def __call__(self, x, is_training=False): 


        x0 = self.c0_0( x, is_training )
        x1 = self.c1_0( x, is_training )
        x2 = self.c2_0( x, is_training )

        x1 = self.c1_1( x1, is_training )

        x2 = self.c2_1( x2, is_training )
        x2 = self.c2_2( x2, is_training )

        xf = concat( [ x0, x1, x2 ], 3, name = "{}_concat".format( self.name )  )

        xf = self.c3_0( xf, is_training )

        x = x + xf

        if self.act:
            x = self.act( x )
        
        print(x)            
        return x

class InceptionResnetBV2Layer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.act = act
        self.name = name
                
        self.c0_0 = Conv2DLayer( 192, 1, 1, '{}_incep_resnet_b_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 128, 1, 1, '{}_incep_resnet_b_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 160, (1,7), 1, '{}_incep_resnet_b_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_2 = Conv2DLayer( 192, (7,1), 1, '{}_incep_resnet_b_1_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 1154, 1, 1, '{}_incep_resnet_b_2_0'.format( name ),
                               dropout, bn, ln, "SAME", None )

                                   
    def __call__(self, x, is_training=False): 


        x0 = self.c0_0( x, is_training )
        x1 = self.c1_0( x, is_training )

        x1 = self.c1_1( x1, is_training )
        x1 = self.c1_2( x1, is_training )

        xf = concat( [ x0, x1 ], 3, name = "{}_concat".format( self.name )  )

        xf = self.c2_0( xf, is_training )

        x = x + xf

        if self.act:
            x = self.act( x )
        
        print(x)            
        return x

class InceptionResnetCV2Layer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.act = act
        self.name = name
                
        self.c0_0 = Conv2DLayer( 192, 1, 1, '{}_incep_resnet_c_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 192, 1, 1, '{}_incep_resnet_c_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 224, (1,3), 1, '{}_incep_resnet_c_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_2 = Conv2DLayer( 256, (3,1), 1, '{}_incep_resnet_c_1_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 2048, 1, 1, '{}_incep_resnet_c_2_0'.format( name ),
                               dropout, bn, ln, "SAME", None )

                                   
    def __call__(self, x, is_training=False): 

        x0 = self.c0_0( x, is_training )
        x1 = self.c1_0( x, is_training )

        x1 = self.c1_1( x1, is_training )
        x1 = self.c1_2( x1, is_training )

        xf = concat( [ x0, x1 ], 3, name = "{}_concat".format( self.name )  )

        xf = self.c2_0( xf, is_training )

        x = x + xf

        if self.act:
            x = self.act( x )
        
        print(x)            
        return x

class InceptionResnetReductionAV2Layer(object):

    def __init__( self,
                  name,
                  k,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.name = name    
        
        self.c0_0 = Conv2DLayer( 385, 3, 2, '{}_incep_resnet_reduct_a_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 128, 1, 1, '{}_incep_resnet_reduct_a_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 224, 3, 1, '{}_incep_resnet_reduct_a_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_2 = Conv2DLayer( 385, 3, 2, '{}_incep_resnet_reduct_a_1_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

                                   
    def __call__(self, x, is_training=False): 

        x0 = maxpool2d( x, 3, 2, padding="SAME" )

        x1 = self.c0_0( x, is_training )
        x2 = self.c1_0( x, is_training )

        x2 = self.c1_1( x2, is_training )
        x2 = self.c1_2( x2, is_training )

        x = concat( [ x0, x1, x2 ], 3, name = "{}_concat".format( self.name )  )
        
        print(x)            
        return x

class InceptionResnetReductionBV2Layer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.name = name
                
        self.c0_0 = Conv2DLayer( 198, 1, 1, '{}_incep_resnet_reduct_b_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )
        self.c0_1 = Conv2DLayer( 266, 3, 2, '{}_incep_resnet_reduct_b_0_1'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 198, 1, 1, '{}_incep_resnet_reduct_b_1_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 264, 3, 2, '{}_incep_resnet_reduct_b_1_1'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 256, 1, 1, '{}_incep_resnet_reduct_b_2_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c2_1 = Conv2DLayer( 288, 3, 1, '{}_incep_resnet_reduct_b_2_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c2_2 = Conv2DLayer( 364, 3, 2, '{}_incep_resnet_reduct_b_2_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

                                   
    def __call__(self, x, is_training=False): 

        x0 = maxpool2d( x, 3, 2, padding="SAME" )

        x1 = self.c0_0( x, is_training )
        x2 = self.c1_0( x, is_training )
        x3 = self.c2_0( x, is_training )

        x1 = self.c0_1( x1, is_training )
        x2 = self.c1_1( x2, is_training )
        x3 = self.c2_1( x3, is_training )
        x3 = self.c2_2( x3, is_training )

        x = concat( [ x0, x1, x2, x3 ], 3, name = "{}_concat".format( self.name )  )
        
        print(x)            
        return x

class InceptionResnetAV2SELayer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.act = act
        self.name = name
                
        self.c0_0 = Conv2DLayer( 32, 1, 1, '{}_incep_resnet_a_0_0'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 32, 1, 1, '{}_incep_resnet_a_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 32, 3, 1, '{}_incep_resnet_a_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 32, 1, 1, '{}_incep_resnet_a_2_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c2_1 = Conv2DLayer( 48, 3, 1, '{}_incep_resnet_a_2_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c2_2 = Conv2DLayer( 64, 3, 1, '{}_incep_resnet_a_2_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c3_0 = Conv2DLayer( 384, 1, 1, '{}_incep_resnet_a_3_0'.format( name ),
                               dropout, bn, ln, "SAME", None )

        self.se = SqueezeNExcitation( '{}_incep_resnet_a_se'.format( name ), act, 16 )

                                   
    def __call__(self, x, is_training=False): 


        x0 = self.c0_0( x, is_training )
        x1 = self.c1_0( x, is_training )
        x2 = self.c2_0( x, is_training )

        x1 = self.c1_1( x1, is_training )

        x2 = self.c2_1( x2, is_training )
        x2 = self.c2_2( x2, is_training )

        xf = concat( [ x0, x1, x2 ], 3, name = "{}_concat".format( self.name ) )

        xf = self.c3_0( xf, is_training )

        xf = self.se( xf, is_training )

        x = x + xf

        if self.act:
            x = self.act( x )
        
        print(x)            
        return x

class InceptionResnetBV2SELayer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.act = act
        self.name = name
                
        self.c0_0 = Conv2DLayer( 192, 1, 1, '{}_incep_resnet_b_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 128, 1, 1, '{}_incep_resnet_b_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 160, (1,7), 1, '{}_incep_resnet_b_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_2 = Conv2DLayer( 192, (7,1), 1, '{}_incep_resnet_b_1_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 1154, 1, 1, '{}_incep_resnet_b_2_0'.format( name ),
                               dropout, bn, ln, "SAME", None )

        self.se = SqueezeNExcitation( '{}_incep_resnet_b_se'.format( name ), act, 16 )

                                   
    def __call__(self, x, is_training=False): 


        x0 = self.c0_0( x, is_training )
        x1 = self.c1_0( x, is_training )

        x1 = self.c1_1( x1, is_training )
        x1 = self.c1_2( x1, is_training )

        xf = concat( [ x0, x1 ], 3, name = "{}_concat".format( self.name )  )

        xf = self.c2_0( xf, is_training )

        xf = self.se( xf, is_training )

        x = x + xf

        if self.act:
            x = self.act( x )
        
        print(x)            
        return x

class InceptionResnetCV2SELayer(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        self.act = act
        self.name = name
                
        self.c0_0 = Conv2DLayer( 192, 1, 1, '{}_incep_resnet_c_0_0'.format( name ), 
                               dropout, bn, ln, "SAME", act )

        self.c1_0 = Conv2DLayer( 192, 1, 1, '{}_incep_resnet_c_1_0'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_1 = Conv2DLayer( 224, (1,3), 1, '{}_incep_resnet_c_1_1'.format( name ),
                               dropout, bn, ln, "SAME", act )
        self.c1_2 = Conv2DLayer( 256, (3,1), 1, '{}_incep_resnet_c_1_2'.format( name ),
                               dropout, bn, ln, "SAME", act )

        self.c2_0 = Conv2DLayer( 2048, 1, 1, '{}_incep_resnet_c_2_0'.format( name ),
                               dropout, bn, ln, "SAME", None )

        self.se = SqueezeNExcitation( '{}_incep_resnet_c_se'.format( name ), act, 16 )

                                   
    def __call__(self, x, is_training=False): 

        x0 = self.c0_0( x, is_training )
        x1 = self.c1_0( x, is_training )

        x1 = self.c1_1( x1, is_training )
        x1 = self.c1_2( x1, is_training )

        xf = concat( [ x0, x1 ], 3, name = "{}_concat".format( self.name )  )

        xf = self.c2_0( xf, is_training )

        xf = self.se( xf, is_training )

        x = x + xf

        if self.act:
            x = self.act( x )
        
        print(x)            
        return x
