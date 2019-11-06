import sys
sys.path.append('../')

import tensorflow as tf

from ann_utils.inception_layer import InceptionV2StemLayer, InceptionResnetAV2Layer, InceptionResnetReductionAV2Layer, \
                                      InceptionResnetBV2Layer, InceptionResnetReductionBV2Layer, InceptionResnetCV2Layer, \
                                      InceptionResnetAV2SELayer, InceptionResnetBV2SELayer, InceptionResnetCV2SELayer 

from ann_utils.models.squeeze_n_excitation import SqueezeNExcitation

from ann_utils.helper import avgpool2d

class InceptionResnetv2SE(object):

    def __init__( self,
                  name,
                  size,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):
                  
        self.steam = InceptionV2StemLayer( name, dropout, bn, ln, act )
        
        self.stack_a = [ InceptionResnetAV2SELayer( "{}_{}_la".format( name, i ), dropout, bn, ln, act )         
                        for i in range(size[0]) ]
        self.reduce_a = InceptionResnetReductionAV2Layer( "{}_reduce_a".format( name ), 512, dropout, bn, ln, act )

        self.stack_b = [ InceptionResnetBV2SELayer( "{}_{}_lb".format( name, i ), dropout, bn, ln, act )
                        for i in range(size[1]) ]
        self.reduce_b = InceptionResnetReductionBV2Layer( "{}_reduce_b".format( name ), dropout, bn, ln, act )

        self.stack_c = [ InceptionResnetCV2SELayer( "{}_{}_lc".format( name, i ), dropout, bn, ln, act )
                        for i in range(size[2]) ]

    def __call__(self, x, is_training=False): 

        x = self.steam( x, is_training )        
        x0 = x

        for l in self.stack_a:
            x = l( x, is_training )
        x1 = x
        x = self.reduce_a( x, is_training )

        for l in self.stack_b:
            x = l( x, is_training )
        x2 = x
        x = self.reduce_b( x, is_training )

        for l in self.stack_c:
            x = l( x, is_training )
        x3 = x

        return x0, x1, x2, x3

class InceptionResnetv2(object):

    def __init__( self,
                  name,
                  size,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):
                  
        self.steam = InceptionV2StemLayer( name, dropout, bn, ln, act )
        
        self.stack_a = [ InceptionResnetAV2Layer( "{}_{}_la".format( name, i ), dropout, bn, ln, \
                                                  act ) \
                        for i in range(size[0]) ]
        self.reduce_a = InceptionResnetReductionAV2Layer( "{}_reduce_a".format( name ), 512, dropout, bn, ln, \
                                                          act )
        
        self.stack_b = [ InceptionResnetBV2Layer( "{}_{}_lb".format( name, i ), dropout, bn, ln, \
                                                  act ) \
                        for i in range(size[1]) ]
        self.reduce_b = InceptionResnetReductionBV2Layer( "{}_reduce_b".format( name ), dropout, bn, ln, \
                                                          act )

        self.stack_c = [ InceptionResnetCV2Layer( "{}_{}_lc".format( name, i ), dropout, bn, ln, \
                                                  act ) \
                        for i in range(size[2]) ]

    def __call__(self, x, is_training=False): 

        x = self.steam( x, is_training )        
        x0 = x

        for l in self.stack_a:
            x = l( x, is_training )
        x1 = x
        x = self.reduce_a( x, is_training )

        for l in self.stack_b:
            x = l( x, is_training )
        x2 = x
        x = self.reduce_b( x, is_training )

        for l in self.stack_c:
            x = l( x, is_training )
        x3 = x

        return x0, x1, x2, x3 