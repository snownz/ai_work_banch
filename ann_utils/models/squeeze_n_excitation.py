import sys
sys.path.append('../../')

import tensorflow as tf

from ann_utils.conv_layer import Conv2DLayer
from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import global_average_pool_spartial
from ann_utils.helper import flatten

class SqueezeNExcitation(object):

    def __init__( self,
                  name,
                  act,
                  ratio=4,
                  dropout=0.0 ):

        self.c1 = Conv2DLayer( None, 1, 1, "cs1", bias = False, act = act )
        self.c2 = Conv2DLayer( None, 1, 1, "cs2", bias = False, act = tf.nn.sigmoid )

        self.ratio = ratio
        self.name = name

    def __call__(self, x, is_training=False):

        with tf.variable_scope(self.name):

            init = x
            se = global_average_pool_spartial( init ) 
            filters = init.shape[-1]
            
            se = self.c1( se, is_training, filters // self.ratio )
            se = self.c2( se, is_training, filters )

            x = init * se

            print( x )
            return x  