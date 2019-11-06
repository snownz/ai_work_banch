import tensorflow as tf
from tensorflow.python.ops import array_ops

from ann_utils.conv_layer import Conv2DLayer, SeparableConv2DLayer
from ann_utils.som_layer import SOMLayer
from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import flatten, l2
from ann_utils.nalu_cell import NaluCell
from ann_utils.nac_cell import NacCell

class MemoryCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, name, memory, state_is_tuple=False, act=tf.nn.relu):

        tf.nn.rnn_cell.RNNCell.__init__(self)

        self.is_training = True

        self._num_units = num_units
        self.memory = memory
        self._state_is_tuple = state_is_tuple

        self.o = FullyLayer( self._num_units, "output_{}".format( name ), act = tf.nn.sigmoid ) 
        self.d = NaluCell( self._num_units, "pred_{}".format( name ), act = act )

        self.fg = FullyLayer( self._num_units, "fg_{}".format( name ), act = tf.nn.sigmoid )
        self.ig = FullyLayer( self._num_units, "ig_{}".format( name ), act = tf.nn.sigmoid )        
        self.mg = FullyLayer( 1, "mg_{}".format( name ), act = tf.nn.sigmoid )        
        
        self.som = memory
        
        self.c1 = Conv2DLayer( 8,  5, 2, "c1_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )
        self.c2 = Conv2DLayer( 16, 3, 2, "c2_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )
        self.c3 = Conv2DLayer( 32, 2, 2, "c3_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )        
        self.c4 = Conv2DLayer( 64, 1, 2, "c4_{}".format( name ), bn = False, kernel_regularizer = l2( 2e-4 ) )
        self.nm = NaluCell( self._num_units, "nm_{}".format( name ), act = act )        

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def set_values(self, is_training):
        self.is_training = is_training

    def __encode(self, x):
        
        x = self.c1( x, is_training = self.is_training )
        x = self.c2( x, is_training = self.is_training )
        x = self.c3( x, is_training = self.is_training )
        x = self.c4( x, is_training = self.is_training )
        x = flatten( x )    

        return self.nm( x, is_training = self.is_training ) 

    def _retrive_memory(self, x, h):
        mems = self.som( x, h )        
        mem = self.__encode( mems )
        return mem
    
    def call(self, x, state):
        
        sm = state

        smx = tf.concat( [ sm, x ], axis = 1 )

        with tf.compat.v1.variable_scope( "logical", reuse = tf.AUTO_REUSE ):
            h = self.d( smx, is_training = self.is_training )
        
        with tf.compat.v1.variable_scope( "intuitive", reuse = tf.AUTO_REUSE ):

            fg = self.fg( smx, is_training = self.is_training )
            ig = self.ig( smx, is_training = self.is_training )
            cg = self._retrive_memory( sm, x )

            sm_ = ( fg * sm ) + ( ( ( 1.0 - fg ) * ig ) * cg )

        with tf.compat.v1.variable_scope( "decision_maker", reuse = tf.AUTO_REUSE ):

            smx_ = tf.concat( [ sm_, x ], axis = 1 )

            o = self.o( smx_, is_training = self.is_training )
            out = tf.nn.tanh( ( o * sm_ ) + h )

        with tf.compat.v1.variable_scope( "intuitive_memory_storage", reuse = tf.AUTO_REUSE ):

            smsm_ = tf.concat( [ sm_, sm ], axis = 1 )
            
            mg = self.mg( smsm_, is_training = self.is_training )

            sm__ = mg * sm_

        return ( out, ( x, sm ) ), sm__