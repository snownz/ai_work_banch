import tensorflow as tf
from tensorflow.python.ops import array_ops

from ann_utils.conv_layer import Conv2DLayer, SeparableConv2DLayer
from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import flatten, l2
from ann_utils.nalu_cell import NaluCell
from ann_utils.nac_cell import NacCell

class MemoryCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, name, memory, act=tf.nn.relu):

        tf.nn.rnn_cell.RNNCell.__init__(self)

        self.num_units = num_units
        self.memory = memory
        self.som = memory

        self.f = FullyLayer( self.num_units, "{}_fg".format( name ), act = tf.nn.sigmoid )
        self.i = FullyLayer( self.num_units, "{}_ig".format( name ), act = tf.nn.sigmoid )  
        self.o = FullyLayer( 1, "{}_ot".format( name ), act = tf.nn.sigmoid ) 
        self.m = FullyLayer( 1, "{}_mg".format( name ), act = tf.nn.sigmoid )      

        self.d = FullyLayer( self.num_units, "{}_pred".format( name ), act = act )
        
        self.c1 = SeparableConv2DLayer( 8,  5, 2, "{}_c1".format( name ), bn = True )
        self.c2 = SeparableConv2DLayer( 16, 3, 2, "{}_c2".format( name ), bn = True )
        self.c3 = SeparableConv2DLayer( 32, 2, 2, "{}_c3".format( name ), bn = True )        
        self.c4 = SeparableConv2DLayer( 64, 1, 2, "{}_c4".format( name ), bn = True )

        self.nm = FullyLayer( self.num_units, "{}_nm".format( name ), act = act )  

        self.is_training = True

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def set_values(self, is_training):
        self.is_training = is_training

    def __encode(self, x):
        
        # x = self.c1( x, is_training = self.is_training )
        # x = self.c2( x, is_training = self.is_training )
        # x = self.c3( x, is_training = self.is_training )
        # x = self.c4( x, is_training = self.is_training )
        x = flatten( x )            
        x = self.nm( x, is_training = self.is_training )        
        return x

    # def _retrive_memory(self, x, h):
    #     mems = self.som( x, h )        
    #     mem = self.__encode( mems )
    #     return mem

    def _retrive_memory(self, x):
        mems = self.som( x )        
        mem = self.__encode( mems )
        return mem
    
    def call(self, x, state):
        
        sm = state
        smx = tf.concat( [ sm, x ], axis = 1 )

        with tf.compat.v1.variable_scope( "logical", reuse = tf.AUTO_REUSE ):
            f = self.f( smx, is_training = self.is_training )
            i = self.i( smx, is_training = self.is_training )
            h = self.d( smx, is_training = self.is_training ) 
            sm_ = ( f * sm ) + ( ( ( 1.0 - f ) * i ) * h )
            logical_out = tf.nn.tanh( sm_ )
        
        with tf.compat.v1.variable_scope( "intuitive", reuse = tf.AUTO_REUSE ):
            # c = self._retrive_memory( sm, x )
            c = self._retrive_memory( smx )
            intuitive_out = tf.nn.tanh( c )
            
        # with tf.compat.v1.variable_scope( "decision_maker", reuse = tf.AUTO_REUSE ):
        #     sm_c = tf.concat( [ sm_, c ], axis = 1 )
        #     o = self.o( sm_c, is_training = self.is_training )
        #     out = tf.nn.tanh( ( o * sm_ ) + c )

        with tf.compat.v1.variable_scope( "memory_storage", reuse = tf.AUTO_REUSE ):
            smsm_ = tf.concat( [ sm_, sm ], axis = 1 )            
            m = self.m( smsm_, is_training = self.is_training )
            sm__ = m * sm_

        return ( ( logical_out, intuitive_out ), ( x, sm ) ), sm__