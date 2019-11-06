import tensorflow as tf
import numpy as np
from random import random

from ann_utils.helper import maxpool2d, avgpool2d, dropout, spectral_norm, norm, bn

class Conv2DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu,
                  bias=True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.bn = bn
        self.ln = ln
        self.name = name 
        self.act = act 
        self.bias = bias

    def __call__(self, x, is_training=False, filters=None): 

        if not filters is None:
            self.output_size = filters

        with tf.variable_scope( self.name ):

            # kernel shape
            k = [ self.kernel, self.kernel, x.shape[-1], self.output_size ] if type(self.kernel) is int \
                else [ self.kernel[0], self.kernel[1], x.shape[-1], self.output_size ]

            # kernel
            w = tf.compat.v1.get_variable( "_w_2d", 
                                        k, 
                                        # initializer = tf.contrib.layers.xavier_initializer(),
                                        initializer = tf.random_normal_initializer( stddev = 0.02 ),
                                        trainable = is_training )
            strides = ( 1, self.stride, self.stride, 1 ) if type(self.stride) is int \
                    else ( 1, self.stride[0], self.stride[1], 1 )

            # setup layer
            x = tf.nn.conv2d( x, w, strides, padding = self.padding )

            if self.bias:
                b = tf.compat.v1.get_variable( '_b_', 
                                            [ 1, 1, 1, self.output_size ], 
                                            initializer = tf.constant_initializer( 0.0 ), 
                                            trainable = is_training )
                x += b
                
            # batch normalization
            if self.bn:
                x = bn( x, is_training = is_training, name = "_bn" )

            # activation
            if not self.act is None:
                x = self.act( x, name = "_act" )

            # setup dropout
            if self.dropout > 0 and is_training:
                x = dropout( x, self.dropout, name = "_dp" )
        
        print(x)
        return x


class NALUConv2DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu,
                  bias=True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.bn = bn
        self.ln = ln
        self.name = name 
        self.act = act 
        self.bias = bias

    def __call__(self, x, is_training=False, filters=None): 

        if not filters is None:
            self.output_size = filters

        with tf.variable_scope( self.name ):

            # kernel shape
            k = [ self.kernel, self.kernel, x.shape[-1], self.output_size ] if type(self.kernel) is int \
                else [ self.kernel[0], self.kernel[1], x.shape[-1], self.output_size ]

            # kernel
            gt = tf.get_variable( "_w_gt_2d", 
                                  k, 
                                  # initializer = tf.contrib.layers.xavier_initializer(),
                                  initializer = tf.truncated_normal_initializer( stddev = 0.02 ),
                                  trainable = is_training )

            wt = tf.get_variable( "_w_wt_2d", 
                                  k, 
                                  # initializer = tf.contrib.layers.xavier_initializer(),
                                  initializer = tf.truncated_normal_initializer( stddev = 0.02 ),
                                  trainable = is_training )

            mt = tf.get_variable( "_w_mt_2d", 
                                  k, 
                                  # initializer = tf.contrib.layers.xavier_initializer(),
                                  initializer = tf.truncated_normal_initializer( stddev = 0.02 ),
                                  trainable = is_training )
            
            strides = ( 1, self.stride, self.stride, 1 ) if type(self.stride) is int \
                    else ( 1, self.stride[0], self.stride[1], 1 )

            with tf.variable_scope( 'nac_w'):
                w = tf.multiply( tf.tanh( wt ), tf.sigmoid( mt ) )

            with tf.variable_scope( 'simple_nac'):
                a = tf.nn.conv2d( x, w, strides, padding = self.padding )
            
            with tf.variable_scope( 'complex_nac' ):
                # m = tf.exp( tf.nn.conv2d( tf.log( tf.abs( x ) + 1e-10 ), w, strides, padding = self.padding ) )
                m = tf.sinh( tf.nn.conv2d( tf.asinh( x ), w, strides, padding = self.padding ) )

            with tf.variable_scope( 'math_gate' ):
                gc = tf.nn.sigmoid( tf.nn.conv2d( x, gt, strides, padding = self.padding ) )

            with tf.variable_scope( 'result' ):
                x = ( gc * a ) + ( ( 1 - gc ) * m )

            # if self.bias:
            #     b = tf.compat.v1.get_variable( '_b_', 
            #                                 [ 1, 1, 1, self.output_size ], 
            #                                 initializer = tf.constant_initializer( 0.0 ), 
            #                                 trainable = is_training )
            #     x += b
                
            # batch normalization
            if self.bn:
                x = bn( x, is_training = is_training, name = "_bn" )

            # activation
            if not self.act is None:
                x = self.act( x, name = "_act" )

            # setup dropout
            if self.dropout > 0 and is_training:
                x = dropout( x, self.dropout, name = "_dp" )
        
        print(x)
        return x

class SNConv2DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu,
                  bias=True
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.name = name 
        self.act = act 
        self.bias = bias

    def __call__(self, x, is_training=False, filters=None): 

        if not filters is None:
            self.output_size = filters

        # kernel shape
        k = [ self.kernel, self.kernel, x.shape[-1], self.output_size ] if type(self.kernel) is int \
            else [ self.kernel[0], self.kernel[1], x.shape[-1], self.output_size ]

        # kernel
        w = tf.compat.v1.get_variable( "{}_w_2d".format( self.name ), 
                                       k, 
                                       initializer = tf.contrib.layers.xavier_initializer(),
                                       trainable = is_training )
        
        w = spectral_norm( w, "{}_2d".format( self.name ) )

        strides = ( 1, self.stride, self.stride, 1 ) if type(self.stride) is int \
                  else ( 1, self.stride[0], self.stride[1], 1 )

        # setup layer
        x = tf.nn.conv2d( x, w, strides, padding = self.padding )

        if self.bias:
            b = tf.compat.v1.get_variable( '{}_b_'.format( self.name ), 
                                           [ 1, 1, 1, self.output_size ], 
                                           initializer = tf.constant_initializer( 0.0 ), 
                                           trainable = is_training )
            x += b
            
        # activation
        if not self.act is None:
            x = self.act( x, name = "{}_act".format(self.name) )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = dropout( x, self.dropout, name = "{}_dp".format(self.name) )
        
        print(x)
        return x

class Deconv2DLayer(Conv2DLayer):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dilation = 1,
                  dropout = 0.0, bn = False, ln = False,
                  padding = "SAME",
                  act = tf.nn.leaky_relu,
                  bias=True
                ):

        super().__init__( output, kernel, stride, name,
                          dropout, bn, ln, padding, act, bias )

        self.dilation = dilation

    def __call__(self, x, is_training=False, filters=None):

        if not filters is None:
            self.output_size = filters

        with tf.variable_scope( self.name ):

            # kernel shape
            k = [ self.kernel, self.kernel, self.output_size, x.shape[-1] ] if type(self.kernel) is int \
                else [ self.kernel[0], self.kernel[1], self.output_size, x.shape[-1] ]

            out_shape = [ tf.shape( x )[0], x.shape[1] * self.stride, x.shape[2] * self.stride, self.output_size ]

            # kernel
            w = tf.compat.v1.get_variable( "_w_2dt", 
                                        k, 
                                        # initializer = tf.contrib.layers.xavier_initializer(),
                                        initializer = tf.random_normal_initializer( stddev = 0.02 ),
                                        # initializer = tf.compat.v1.initializers.identity(),
                                        trainable = is_training )
            strides = ( 1, self.stride, self.stride, 1 ) if type(self.stride) is int \
                    else ( 1, self.stride[0], self.stride[1], 1 )

            # setup layer
            x = tf.nn.conv2d_transpose( x, w, out_shape, strides, padding = self.padding )

            if self.bias:
                b = tf.compat.v1.get_variable('_b_', [ self.output_size ], initializer = tf.constant_initializer( 0 ), trainable = is_training )
                x = tf.nn.bias_add( x, b, name = "_bias" )
                
            # batch normalization
            if self.bn:
                x = bn( x, is_training = is_training, name = "_bn" )

            # activation
            if not self.act is None:
                x = self.act( x, name = "_act" )

            # setup dropout
            if self.dropout > 0 and is_training:
                x = dropout( x, self.dropout, name = "_dp" )
        
        print(x)
        return x

class SeparableConv2DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name,
                  dropout = 0.0, bn = False, ln = False,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu
                ):

        self.output_size = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.bn = bn 
        self.ln = ln 
        self.name = name 
        self.act = act 

    def __call__(self, x, is_training=False): 

        # setup layer
        x = tf.layers.separable_conv2d( x,
                                        filters            = self.output_size,
                                        bias_initializer   = tf.zeros_initializer(),
                                        kernel_size        = [ self.kernel, self.kernel ] if type(self.kernel) is type(int) else self.kernel, 
                                        strides            = [ self.stride, self.stride ],
                                        padding            = self.padding,
                                        trainable          = is_training,
                                        name               = self.name
                                      )

         # batch normalization
        if self.bn:
            x = bn( x, is_training = is_training, name = "{}_bn_".format( self.name ) )

        if self.ln:
            # x = norm( x, self.name )
            x = ln( x, is_training = is_training )

        # activation
        if not self.act is None:
            x = self.act( x, name = "{}_act".format(self.name) )

        # setup dropout
        if self.dropout > 0 and is_training:
            x = dropout( x, self.dropout, name = "{}_dp".format(self.name) )
        
        print(x)           
        return x

class Conv1DLayer(object):

    def __init__( self, 
                  output, 
                  kernel, stride, 
                  name, 
                  dropout = 0.0,
                  padding = "SAME",  
                  act = tf.nn.leaky_relu
                ):

        self.nf = output 
        self.dropout = dropout 
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.name = name 
        self.act = act 

    def __call__(self, x, is_training=False): 

        with tf.variable_scope( self.name ):
            # setup layer
            nx = x.shape.as_list()[2]
            
            w = tf.compat.v1.get_variable( '_w_', 
                                        [ self.kernel, nx, self.nf ], 
                                        initializer = tf.random_normal_initializer( stddev = 0.02 ) )
            b = tf.compat.v1.get_variable('_b_', [ self.nf ], initializer = tf.constant_initializer(0) )
            
            strides = [ self.stride, self.stride, 1 ] if type(self.stride) is int \
                    else [ 1, self.stride[0], self.stride[1] ]

            x = tf.nn.conv1d( x, w, stride = strides, padding = self.padding ) + b
            
            # activation
            if not self.act is None:
                x = self.act( x )

            # setup dropout
            if self.dropout > 0 and is_training:
                x = tf.layers.dropout( inputs = x, rate = self.dropout )
            
            print(x)
            return x