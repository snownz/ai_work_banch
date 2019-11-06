import tensorflow as tf
import numpy as np

from ann_utils.helper import flatten, maxpool2d,\
                             hw_flatten, hw_flatten_multi_head,\
                             softmax, upsampling2d, get_median,\
                             to_float, norm,\
                             shape_list, upsampling1d

from ann_utils.conv_layer import Conv2DLayer, SNConv2DLayer, Conv1DLayer
from ann_utils.fully_layer import FullyLayer as key

"""
    Self Attention for Image Inputs
    matmul( [ m1, n1 ], [ m2, n2 ] ) = [ m1, n2 ]
    matmul( [ m1, n1, c1 ], [ m2, n2, c1 ] ) = [ m1, n2 ]
"""
class Self_Attention_Multi_Head_3D_GB(object):

    def __init__(self, ch, key_size=1, heads=8, dp=0.0, bn=False, act=None, out_act=None):
        
        self.key   = [ Conv2DLayer( key_size, 1, 1, "attn_head_f_conv_layer_{}".format( x ), act = act ) for x in range( heads ) ]
        self.query = [ Conv2DLayer( key_size, 1, 1, "attn_head_g_conv_layer_{}".format( x ), act = act ) for x in range( heads ) ]
        self.value = [ Conv2DLayer( key_size, 1, 1, "attn_head_h_conv_layer_{}".format( x ), act = act ) for x in range( heads ) ]
        self.oc = Conv2DLayer( ch, 1, 1, "attn_head_o_conv_layer", act = out_act )

    def _qkv_(self, x, q_op, k_op, v_op, summary, is_training, reduction):
    
        k = k_op( x, is_training = is_training ) # [bs, h, w, c'] Key
        q = q_op( x, is_training = is_training ) # [bs, h, w, c'] Query
        v = v_op( x, is_training = is_training ) # [bs, h, w, c] Value

        k = maxpool2d( k, 2, 2 )
        v = maxpool2d( v, 2, 2 )
        
        if summary:

            tf.summary.image( '1_query', q, max_outputs = 1, family = "self_attention" )
            tf.summary.image( '2_key', k, max_outputs = 1, family = "self_attention" )
            tf.summary.image( '3_value', v, max_outputs = 1, family = "self_attention" )
           
        return q, k ,v

    def __call__( self, x, is_training=False, summary=False, reduction=1):
        
        with tf.variable_scope('attn'):
            
            with tf.variable_scope('reduction_dim'):
                x = maxpool2d( x, reduction, reduction )
                batch_size = tf.shape(x)[0]
                height = x.shape[1]
                width = x.shape[2]
                ch = x.shape[3]

            with tf.variable_scope('q_k_v'):
                # [ [ batch, h, w, c ] ]
                qkv = [ self._qkv_( x, q, k, v, summary, is_training, reduction ) 
                        for q, k, v in zip( self.key, self.query, self.value ) ]

            with tf.variable_scope('join_heads'):
                # [ batch, heads, h, w, c ]
                qs = tf.concat( [ tf.expand_dims( vl[0], axis = 1 ) for vl in qkv ], axis = 1 )
                ks = tf.concat( [ tf.expand_dims( vl[1], axis = 1 ) for vl in qkv ], axis = 1 )
                vs = tf.concat( [ tf.expand_dims( vl[2], axis = 1 ) for vl in qkv ], axis = 1 )

            with tf.variable_scope('scaled_dop_product'):
                # [ batch, heads, h * w, c ]
                w, s, a = multihead_attn( qs, ks, vs )

            with tf.variable_scope('merge_heads'):
                # [ batch, h * w, c * heads ]
                merged = merge_heads( tf.transpose( a, [0, 2, 3, 1] ) )
                # a = tf.reshape( a, [ batch_size, a.shape[1], height, width, ch ] )
                # merged_image = tf.reduce_mean( a, axis = 1 )

                # [ batch, h, w, c * heads ]
                merged_image = tf.reshape( merged, [ batch_size, height, width, merged.shape[-1] ] )

            with tf.variable_scope('output_attention'):
                # [ batch, h, w, c ]
                o = self.oc( merged_image, is_training = is_training )

            with tf.variable_scope('restore_dim'):
                # [ batch, h, w, c ]
                attn = upsampling2d( o, reduction )
        
        return attn

"""
    Self Attention for Sequences Inputs
"""
class Self_Attention_Multi_Head_2D_GB(object):

    def __init__(self, n_state, name, heads=8, dp=0.0, act=None, out_act=None):

        self.name = name
        self.out_act = out_act
        self.heads = heads
        self.n_state = n_state
        
        self.c = [ Conv1DLayer( n_state * 3, 1, 1, '{}_c_attn_{}'.format(name, x) ) for x in range( heads ) ]
        self.o = Conv1DLayer( n_state, 1, 1, '{}_o_attn'.format(name) )

    def _qkv_(self, x, op, summary, is_training):

        c = op( x, is_training )
        q, k, v = tf.split( c, 3, axis = 2 )                   
        return q, k ,v

    def __call__( self, x, use_mask=False, past=None, is_training=False, summary=False):
        
        with tf.compat.v1.variable_scope('_attn_'):

            qkv = [ self._qkv_( x, c, summary, is_training ) for c in self.c ]

            qs = tf.concat( [ tf.expand_dims( vl[0], axis = 1 ) for vl in qkv ], axis = 1 )
            ks = tf.concat( [ tf.expand_dims( vl[1], axis = 1 ) for vl in qkv ], axis = 1 )
            vs = tf.concat( [ tf.expand_dims( vl[2], axis = 1 ) for vl in qkv ], axis = 1 )

            present = tf.stack( [ ks, vs ], axis = 1 )
            if past is not None:
                pk, pv = tf.unstack( past, axis = 1 )
                ks = tf.concat( [ pk, ks ], axis =- 2 )
                vs = tf.concat( [ pv, vs ], axis =- 2 )

            if use_mask:
                w, s, a = masked_multihead_attn( qs, ks, vs )
            else:
                w, s, a = multihead_attn( qs, ks, vs )
                
            a = merge_heads( tf.transpose( a, [0, 2, 3, 1] ) )
            o = self.o( a, is_training )

            # [ batch, h, w, c ]
            attn = o
            
        vars = [ x for x in tf.compat.v1.trainable_variables() if "{}_attn_".format( self.name ) in x.name ]
        
        if summary:
            
            for w in vars:
                tf.summary.histogram( family = 'self_attention', name = w.name, values = w )
            
        return attn, present, vars

def multihead_attn(q, k, v):

    if len( q.shape ) == 5 and len( k.shape ) == 5 and len( v.shape ) == 5:
        # N = h * w
        q = hw_flatten_multi_head( q ) # [ bs, N, c ]
        k = hw_flatten_multi_head( k ) # [ bs, N, c ]
        v = hw_flatten_multi_head( v ) # [ bs, N, c ]
        
    # q, k, v have shape [ batch, heads, ... ]
    w = tf.matmul( q, k, transpose_b = True )

    # divide by sqrt to keep stable gradients
    w = w * tf.rsqrt( tf.cast( v.shape[-1].value, w.dtype ) )
    w = ( w - tf.reduce_min( w ) ) / ( tf.reduce_max( w ) - tf.reduce_min( w ) )

    s = softmax( w )
    a = tf.matmul( s, v )
    return w, s, a

def masked_multihead_attn(q, k, v):

    # q, k, v have shape [ batch, heads, sequence, features ]
    w = tf.matmul( q, k, transpose_b = True )

    # divide by sqrt to keep stable gradients
    w = w * tf.rsqrt( tf.cast( v.shape[-1].value, w.dtype ) )

    w = mask_attn_weights( w )
    s = softmax( w )
    a = tf.matmul( s, v )
    return w, s, a

def mask_attn_weights(w):

    # w [ batch, heads, dst_sequence, src_sequence ], where information flows from src to dst.
    _, _, nd, ns = shape_list( w )
    
    b = attention_mask( nd, ns, dtype = w.dtype )
    b = tf.reshape( b, [ 1, 1, nd, ns ] )
    
    w = w * b - tf.cast( 1e10, w.dtype ) * ( 1 - b )
    return w

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range( nd )[:,None]
    j = tf.range( ns )
    m = i >= j - ns + nd
    return tf.cast( m, dtype )

def merge_heads(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape( x, start + [ a * b ] )

def split_heads(x, n_head):
    # From [batch, sequence, features] to [batch, heads, sequence, features]
    return tf.transpose( split_states( x, n_head ), [ 0, 2, 1, 3 ] )

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

# class Self_Attention_Multi_Head_3D_GB(object):

#     def __init__(self, ch, name, heads = 8, dp=0.0, bn=False, act=None, out_act=None):

#         self.name = name
#         self.out_act = out_act
        
#         self.key = [ Conv2DLayer( ch, 1, 1, "{}_attn_head_{}_f_conv".format( name, x ), dropout = dp, bn = bn, act = act ) for x in range( heads ) ]
#         self.query = [ Conv2DLayer( ch, 1, 1, "{}_attn_head_{}_g_conv".format( name, x ), dropout = dp, bn = bn, act = act ) for x in range( heads ) ]
#         self.value = [ Conv2DLayer( ch, 1, 1, "{}_attn_head_{}_h_conv".format( name, x ), dropout = dp, bn = bn, act = act ) for x in range( heads ) ]

#     def __create_net(self, x, q_op, k_op, v_op, summary, is_training, reduction):

#         feat = maxpool2d( x, reduction, reduction )

#         batch_size = tf.shape(feat)[0]
#         height = feat.shape[1]
#         width = feat.shape[2]
#         num_channels = feat.shape[3]
    
#         k = k_op( feat, is_training = is_training ) # [bs, h, w, c'] Key
#         q = q_op( feat, is_training = is_training ) # [bs, h, w, c'] Query
#         v = v_op( feat, is_training = is_training ) # [bs, h, w, c] Value

#         k = maxpool2d( k, 2, 2 )
#         v = maxpool2d( v, 2, 2 )

#         # N = h * w
#         qf = hw_flatten( q ) # [ bs, N, c ]
#         kf = hw_flatten( k ) # [ bs, N, c ]

#         s = tf.matmul( qf, kf, transpose_b = True ) # [ bs, Ng, Nf ]

#         # sf = flatten( s )
#         # beta = softmax( tf.reshape( sf, tf.shape( s ) ), 1 ) # attention map
#         beta = softmax( s , 2 ) # attention map

#         vf = hw_flatten( v ) # [ bs, N, c ]
#         o = tf.matmul( beta, vf )  # [ bs, N, C ]
#         mask = tf.reshape( o, [ batch_size, height, width, num_channels ] )
        
#         if summary:

#             tf.summary.image( '0_input', x, max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '2_query', q, max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '3_key', k, max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '4_value', v, max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '5_mask', mask, max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '6_scores', tf.image.resize( tf.expand_dims( s, axis = 3 ), [ 32, 32 ] ), max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '7_probs', tf.image.resize( tf.expand_dims( beta, axis = 3 ), [ 32, 32 ] ), max_outputs = 1, family = "self_attention" )
        
#         return mask

#     def __call__( self, x, is_training=False, summary=False, reduction=1):
        
#         masks = [ self.__create_net( x, q, k, v, summary, is_training, reduction ) for q, k, v in zip( self.key, self.query, self.value ) ]

#         msks = tf.concat( masks, axis = 3 )

#         o = self.oc( msks, is_training = is_training )
    
#         gamma = tf.get_variable( "{}_attn_gamma".format( self.name ), [1], 
#                                 initializer = tf.constant_initializer(0.0), 
#                                 trainable = is_training )

#         attn = upsampling2d( gamma * o, reduction ) + x
#         # attn = norm( attn, "{}_attn_".format( self.name ), is_training = is_training )
#         attn = ln( ln )

#         vars = [ x for x in tf.compat.v1.trainable_variables() if "{}_attn_".format( self.name ) in x.name ]
        
#         if summary:

#             tf.summary.image( '8_mask_c', o, max_outputs = 1, family = "self_attention" )
#             tf.summary.image( '9_attn', attn, max_outputs = 1, family = "self_attention" )

#             for w in vars:
#                 tf.summary.histogram( family = 'self_attention', name = w.name, values = w )
            
#         return attn, vars

# def split_heads_2D(x, n_head):
#     # From [batch, sequence, features] to [batch, heads, sequence, features]
#     return tf.transpose( split_states( x, n_head ), [ 0, 2, 1, 3 ] )

# def split_heads_3D(x):
#     # From [ batch, h, w, c ] to [ batch, heads , h, w, c ]
#     return tf.transpose( split_states( x, n_head ), [ 0, 2, 1, 3 ] )

# def split_states(x, n):
#     """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
#     *start, m = shape_list(x)
#     return tf.reshape( x, start + [ n, m // n ] )