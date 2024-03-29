import tensorflow as tf
from tensorflow.python.ops import array_ops

from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import flatten
from ann_utils.nalu_cell import NaluCell
from ann_utils.nac_cell import NacCell

class NaluLSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, name, state_is_tuple = True):

        tf.nn.rnn_cell.RNNCell.__init__(self)

        self.reuse = False
        self.is_training = True
        self._state_is_tuple = state_is_tuple

        self._num_units = num_units

        self.f = FullyLayer( self._num_units, "forget_{}".format( name ), act = tf.nn.sigmoid ) # forget memory
        self.c = FullyLayer( self._num_units, "cell_{}".format( name ),   act = tf.nn.tanh    ) # add memory
        self.i = FullyLayer( self._num_units, "ignore_{}".format( name ), act = tf.nn.sigmoid ) # ignore prediction
        self.o = FullyLayer( self._num_units, "output_{}".format( name ), act = None ) # output prediction

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._num_units

    def set_values(self, reuse, is_training):
        self.reuse = reuse
        self.is_training = is_training

    def call(self, inputs, state):
        
        if type(state) is tuple:
            c_prior = state[0]
            h = state[1]
        else:
            c_prior = array_ops.slice( state, [ 0, 0 ], [ -1, self._num_units ] )
            h = array_ops.slice( state, [ 0, self._num_units ], [ -1, self._num_units ] )

        hx = tf.concat( [ h, inputs ], axis = 1 )
        hx = tf.nn.dropout( hx, 0.5 )
        
        # gates
        f = self.f( hx, is_training = self.is_training )
        i = self.i( hx, is_training = self.is_training )
        o = self.o( hx, is_training = self.is_training )
        c = self.c( hx, is_training = self.is_training )

        # cell and hidden
        _c = ( f * c_prior ) + ( i * c )
        _h = o * tf.nn.tanh( _c )

        return _h, ( ( _h, _c ) if self._state_is_tuple else array_ops.concat([ _h, _c ], 1) )
