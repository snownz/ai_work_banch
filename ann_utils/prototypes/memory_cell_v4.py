import tensorflow as tf
from tensorflow.python.ops import array_ops

from ann_utils.fully_layer import FullyLayer
from ann_utils.helper import flatten
from ann_utils.nalu_cell import NaluCell
from ann_utils.nac_cell import NacCell

class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, input_size, num_units, mem_size, name, state_is_tuple = True):

        tf.nn.rnn_cell.RNNCell.__init__(self)

        self.reuse = False
        self.is_training = True
        self._state_is_tuple = state_is_tuple

        self._num_units = num_units
        self.mem_size = mem_size

        self.f = FullyLayer( num_units, "forget_{}".format( name ), act = tf.nn.sigmoid )
        self.c = FullyLayer( num_units, "cell_{}".format( name ),   act = tf.nn.tanh    )
        self.i = FullyLayer( num_units, "ignore_{}".format( name ), act = tf.nn.sigmoid )
        self.i = FullyLayer( num_units, "output_{}".format( name ), act = tf.nn.sigmoid )
        
        self.input_attn = FullyLayer( input_size, "input_attn{}".format( name ), act = tf.nn.softmax )
        self.cx_emdeding = FullyLayer( num_units, "cx_emdeding{}".format( name ), act = None )

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def set_values(self, reuse, is_training):
        self.reuse = reuse
        self.is_training = is_training

    #state [ batch, value ]
    #context [ batch, value ]
    # mem [ l, c ]
    def call(self, state, context):
                
        cx = tf.concat( [ context, state ], axis = 1 )

        i_attn_mask = self.input_attn( cx, self.is_training )

        x = i_attn_mask * state

        cx_relationship = self.cx_emdeding( cx, self.is_training )

        norm_embeding = tf.nn.l2_normalize( a, axis = 1 )
        mem_embeding = tf.nn.l2_normalize( self.mem, axis = 0 )

        cosine_similarity = dot( a, b ) / ( norm(a)*norm(b))
        
        # gates
        f = self.f( hx, reuse = self.reuse, is_training = self.is_training )
        i = self.i( hx, reuse = self.reuse, is_training = self.is_training )
        o = self.o( hx, reuse = self.reuse, is_training = self.is_training )
        c = self.c( hx, reuse = self.reuse, is_training = self.is_training )

        # cell and hidden
        _c = ( f * c_prior ) + ( i * c )
        _h = o * tf.nn.tanh( _c )

        return _h, ( ( _h, _c ) if self._state_is_tuple else array_ops.concat([ _h, _c ], 1) )
