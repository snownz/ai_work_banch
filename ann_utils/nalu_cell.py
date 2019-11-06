import tensorflow as tf
import numpy as np

from ann_utils.helper import dropout
from ann_utils.nac_cell import NacCell

class NaluCell(object):

    def __init__( self, 
                  output,  
                  name, 
                  dropout = 0.0, 
                  l1 = 0.0, l2 = 0.0, 
                  act = None, 
                  trainable = True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.name = name 
        self.l1 = l1 
        self.l2 = l2 
        self.act = act 
        self.trainable = trainable

        self._add_sub_nac = NacCell( output, "add_subtract_NAC_{}".format(name) )
        self._mult_div_nac = NacCell( output, "multiply_divide_NAC_{}".format(name) )

    def __call__(self, input, reuse=False, is_training=False): 

        with tf.variable_scope( self.name ):
            # setup layer
            x = input
            input_size = input.shape[1].value 

            g = tf.get_variable( "_w_g_", 
                                [ input_size, self.output_size ], 
                                dtype = tf.float32, 
                                initializer = tf.truncated_normal_initializer( stddev = .01 ),
                                trainable = is_training
                            )
            
            wt = tf.get_variable( "_w_wt_", 
                                [ input_size, self.output_size ], 
                                dtype = tf.float32, 
                                initializer = tf.truncated_normal_initializer( stddev = .01 ),
                                trainable = is_training
                                )     

            mt = tf.get_variable( "_w_mt_", 
                                [ input_size, self.output_size ], 
                                dtype = tf.float32, 
                                initializer = tf.truncated_normal_initializer( stddev = .01 ),
                                trainable = is_training
                                )
            
            with tf.variable_scope( 'nac_w'):
                w = tf.multiply( tf.tanh( wt ), tf.sigmoid( mt ) )

            with tf.variable_scope( 'simple_nac'):
                a = tf.matmul( x, w )
            
            with tf.variable_scope( 'complex_nac' ):
                # m  = tf.exp( self._mult_div_nac( tf.log( tf.abs( x ) + 1e-10 ) ) )
                m = tf.sinh( tf.matmul( tf.asinh( x ), w ) )

            with tf.variable_scope( 'math_gate' ):
                gc = tf.sigmoid( tf.matmul( x, g ) )

            with tf.variable_scope( 'result' ):
                x  = ( gc * a ) + ( ( 1 - gc ) * m )
            
            # activation
            if not self.act is None:
                x = self.act( x )

            # setup dropout
            if self.dropout > 0 and is_training:
                x = tf.layers.dropout( inputs = x, rate = self.dropout )

            if not reuse: self.layer = x        
            
            print(x)            
            return x