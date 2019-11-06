import tensorflow as tf
import numpy as np

from ann_utils.helper import flatten, maxpool2d,\
                             hw_flatten, hw_flatten_multi_head,\
                             softmax, upsampling2d, get_median,\
                             to_float, norm,\
                             shape_list, gelu, lrelu, prelu

from ann_utils.conv_layer import Conv2DLayer, SNConv2DLayer, Conv1DLayer
from ann_utils.fully_layer import FullyLayer as fc
from ann_utils.self_attention import Self_Attention_Multi_Head_2D_GB as att

class Transformer(object):

    def __init__(self, state_size, output_embeding, act=None, n_layers=1, num_heads=8 ):
        
        self.state_size = state_size
        self.num_heads = num_heads
        self.output_embeding = output_embeding
        self.n_layers = n_layers
        self.act = act
        self.attn_masked = att( output_embeding, 'm_attn', heads = self.num_heads )
        
    def __call__(self, x, y, is_training=False, summary=False):

        with tf.compat.v1.variable_scope('transformer_encode'):
            #kvs = []
            for l in range( self.n_layers ):
                x, kv = self._build_encoding( x, str(l), is_training, summary )
                #kvs.append( kv )

        with tf.compat.v1.variable_scope('transformer_decode_base'):
            xdb = self._build_decoding_base( y, 'y', is_training, summary )

        with tf.compat.v1.variable_scope('transformer_decode'):
            for l in range( self.n_layers ):
                xd = self._build_decoding( xdb, kv, str(l), is_training, summary )

        batch, sequence, embeding = shape_list( xd )
        x = tf.reshape( xd, [ batch, sequence * embeding ] )

        return x

    def _build_encoding(self, x, name, is_training, summary):

        with tf.compat.v1.variable_scope( name ):

            if self.state_size != x.shape[-1]:
                x = Conv1DLayer( self.state_size, 1, 1, 'resample' )( x, is_training )

            attn, kv, _ = att( self.state_size, 'e_attn', heads = self.num_heads )( x, False, None, is_training, summary )
            x_norm = norm( x + attn, "e_norm", is_training = is_training )
            x = Conv1DLayer( self.state_size, 1, 1, 'ec', act = self.act )( x_norm, is_training )
            return x_norm + x, kv
    
    def _build_decoding_base(self, x, name, is_training, summary):

        with tf.compat.v1.variable_scope(name):

            attn, _, _ = self.attn_masked( x, True, None, is_training, summary )
            x_norm = norm( x + attn, "d_norm", is_training = is_training )

        return x_norm

    def _build_decoding(self, x, past, name, is_training, summary):

        with tf.compat.v1.variable_scope(name):

            if self.state_size != x.shape[-1]:
                x = Conv1DLayer( self.state_size, 1, 1, 'resample' )( x, is_training )

            attn, _, _ = att( self.state_size, 'd_s_attn', heads = self.num_heads )( x, False, None, is_training, summary )
            x_norm = norm( x + attn, "d_norm", is_training = is_training )

            attn, _, _ = att( self.state_size, 'd_ed_attn', heads = self.num_heads )( x, False, past, is_training, summary )
            x_norm = norm( x + attn, "d_norm1", is_training = is_training )

            x = Conv1DLayer( self.state_size, 1, 1, 'dc', act = self.act )( attn, is_training )

        return x_norm + x

"""
From OpenAI
"""         
class GPT2(object):

    def __init__(self, n_ctx=1024, n_head=4, n_layer=4):
        
        self.n_ctx = n_ctx
        self.n_head = n_head
        self.n_layer = n_layer
            
    def __call__(self, x, past, is_training=False, summary=False):

        with tf.compat.v1.variable_scope( 'gpt2_rl' ):

            if past.shape[-1] != x.shape[-1]:
                x = Conv1DLayer( past.shape[-1], 1, 1, 'resample' )( x, is_training )
                
            batch, sequence, n_size = shape_list( x )

            wte = tf.get_variable( 'wte', [ self.n_ctx, sequence * n_size ],
                                   initializer = tf.random_normal_initializer( stddev = 0.02 ) )
            
            h = x
            presents = []
            pasts = tf.unstack( past, axis = 1 ) if past is not None else [None] * self.n_layer

            for layer, past in enumerate(pasts):
                h, present = block( h, 'h%d' % layer, past, n_size, self.n_head, is_training, summary )
                presents.append( present )
            
            presents = tf.stack( presents, axis = 1 )
            h = norm( h, 'ln_f' )

            _, _, n_size = shape_list( h )
            h_flat = tf.reshape( h, [ batch, n_size * sequence ] )

            logits = tf.nn.relu( tf.matmul( h_flat, wte, transpose_b = True ) )

            return logits, presents

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor( value, name = 'value' )
    ndims = value.shape.ndims
    return tf.tile( tf.expand_dims( value, axis = 0 ), [ size ] + [ 1 ] * ndims )

def positions_for(tokens, past_length):
    batch_size = tf.shape( tokens )[0]
    nsteps = tf.shape( tokens )[1]
    return expand_tile( past_length + tf.range( nsteps ), batch_size )

def mlp(x, n_state, is_training):

    nx = x.shape[-1].value
    h = tf.nn.relu( Conv1DLayer( n_state, 1, 1, 'c_fc' )( x, is_training ) )
    h2 = Conv1DLayer( nx, 1, 1, 'c_proj' )( h, is_training )
    
    return h2

def block(x, name, past, n_e, num_heads, is_training=False, summary=False):
    
    with tf.compat.v1.variable_scope( name ):

        if summary:
            tf.summary.image( family = 'gpt', name = '{}_in'.format( name ), tensor = tf.image.resize_bicubic( x[:,:,:,tf.newaxis], [ x.shape[-1], x.shape[-1] ] ), max_outputs = 1 )

        nx = x.shape[-1].value
        norm_x = norm( x, 'ln_1', is_training = is_training )
        a, present, _ = att( n_e, '_attn_', heads = num_heads )( norm_x, True, past, is_training, summary )
        x = x + a
        m = mlp( norm( x, 'ln_2' ), nx * 4, is_training )
        x = x + m

        if summary:
            tf.summary.image( family = 'gpt', name = '{}_out'.format( name ), tensor = tf.image.resize_bicubic( x[:,:,:,tf.newaxis], [ x.shape[-1], x.shape[-1] ] ), max_outputs = 1 )

    return x, present