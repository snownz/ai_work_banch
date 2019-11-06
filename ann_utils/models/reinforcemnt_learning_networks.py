import sys
sys.path.append('../../')

import tensorflow as tf
import numpy as np

from ann_utils.fully_layer import FullyLayer as fc
from ann_utils.fully_layer import mlp

from ann_utils.prototypes.lstm_nac_cell import NaluLSTMCell
from ann_utils.models.transformer import Transformer

from ann_utils.helper import gelu, prelu, mse_loss, to_float, noise, categorical_sample, softmax, gelu

class LSTM_Policy(object):

    def __init__(self,
                 size=256,
                 dp=0.25,
                 act=None,
                 initial_learning_rate=.1,
                 decay_steps=1000,
                 decay_keep=.90):

        self.size = size
        self.act = act
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_keep = decay_keep

            
    def __call__(self, x, is_training=False, summary=False, state_in=None, name=''):

        with tf.compat.v1.variable_scope('lstm_policy'):

            # lstm = tf.nn.rnn_cell.LSTMCell( self.size, state_is_tuple = False, name = name )
            lstm = NaluLSTMCell( self.size, state_is_tuple = False, name = name )

            # introduce a "fake" batch dimension of 1 to do LSTM over time dim
            x = tf.expand_dims( x, [0] )
            
            # state_in = tf.nn.rnn_cell.LSTMStateTuple( c_in, h_in )            
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn( lstm, x, initial_state = state_in, time_major = True )
            lstm_c, lstm_h = tf.split( lstm_state, 2, axis = 1 )

            x = tf.reshape( lstm_outputs, [ -1, self.size ] )

            state_out = [ lstm_c[:1, :], lstm_h[:1, :] ]
            
        return x, state_out























































    # xs = tf.concat( [ i_s, x ], axis = 1 )
    # simi, _, retrived = self.__retrieve_memory_learn_function( xs, True )
    
    # xq = tf.concat( [ x, retrived ], axis  = 1 )
    # xq = x + retrived

    # def __retrieve_memory_learn_function(self, x, is_training=False):
        
    #     # transform memory to 3D [ 1, m, h, 1 ]
    #     batch_memory = tf.expand_dims( self.mem, axis = 0 )
        
    #     cm = conv( 16, 3, 2, "{}_memory_conv1".format( self.name ), dropout = 0.25, bn = True, act = self.hidden_act )( batch_memory, is_training )
    #     cm = conv( 16, 3, 2, "{}_memory_conv2".format( self.name ), dropout = 0.25, bn = True, act = self.hidden_act )( cm, is_training )
    #     cm = conv( 32, 3, 2, "{}_memory_conv3".format( self.name ), dropout = 0.25, bn = True, act = self.hidden_act )( cm, is_training )
    #     cm = conv( 64, 2, 2, "{}_memory_conv4".format( self.name ), dropout = 0.25, bn = True, act = self.hidden_act )( cm, is_training )
    #     cm = conv( 128, 2, 2, "{}_memory_conv5".format( self.name ), dropout = 0.25, bn = True, act = self.hidden_act )( cm, is_training )
       
    #     cm = flatten( cm )
    #     cm = tf.reduce_mean( cm, axis = 0 )
    #     cm = tf.reshape( 
    #         tf.tile( 
    #             cm, 
    #             [ tf.shape( x )[ 0 ] ] ), 
    #         [ tf.shape( x )[ 0 ], tf.shape( cm )[ 0 ] ] )

    #     # compute similarity function
    #     x_simi = tf.concat( [ x, cm ], axis = 1 )
    #     simi = fc( self.mem_size, 
    #                '{}_memory_similarity'.format( self.name ), 
    #                act = None )( x_simi, is_training )   
                
    #     # compute scores
    #     scores = softmax( simi )
        
    #     # retive memory info
    #     retrived = tf.matmul( 
    #         scores, 
    #         tf.reshape( 
    #             self.mem, 
    #             [ self.mem.shape[0], self.mem.shape[1] ] ) )
      
    #     return simi, scores, retrived

    # def _build_memory_update(self, irsa):

    #     # with tf.compat.v1.variable_scope("mem_update"):
                       
    #     #     batch_size = tf.shape( self.pre_memory )[0]
    #     #     encode_size = self.state_size + self.h_size[-1]

    #     #     # tranform features to 2D
    #     #     to_write = tf.tile( self.pre_memory, tf.stack( [ 1, self.mem_size ] ), name = "to_write_tile" )
    #     #     to_write = tf.reshape( to_write, [ batch_size, self.mem_size, encode_size ], name = "to_write_reshape" )

    #     #     # normalize instrinsic reward
    #     #     irsa_ = 1.0 - ( ( irsa - tf.reduce_min( irsa ) ) / ( tf.reduce_max( irsa ) - tf.reduce_min( irsa ) ) )
            
    #     #     # invert values ... 1 is a visited state ... 0 unvisited      
    #     #     new_m_factor = tf.reshape( tf.tile( irsa_, [ self.mem_size ] ), [ batch_size, self.mem_size ] )           
            
    #     #     # priorize unvisited states
    #     #     score_w = self.mem_simi + new_m_factor
    #     #     score_w = ( ( score_w - tf.reduce_min( score_w ) ) / ( tf.reduce_max( score_w ) - tf.reduce_min( score_w ) ) )

    #     #     # compute scores with noise, to avoid inicialization bias
    #     #     scores_r = softmax( score_w + ( self.memory_lr * noise( score_w ) ) )
            
    #     #     # select % of memory positions
    #     #     # bin_socres = to_float( scores_r > ( 1.0 + self.mem_beta ) * tf.reduce_mean( scores_r, axis = 1, keepdims = True ) )
    #     #     bin_socres = to_float( scores_r >= tf.reduce_max( scores_r, axis = 1, keepdims = True ) )
    #     #     bin_socres_p = bin_socres * scores_r
            
    #     #     # reshape scores
    #     #     expand_scores = tf.tile( bin_socres_p, [ 1, encode_size ], name = "expand_scores_tile" )
    #     #     new_scores = tf.reshape( expand_scores, [ batch_size, encode_size, self.mem_size ], name = "new_scores_reshape" )
    #     #     to_write_scores = tf.transpose( new_scores, perm = [ 0, 2, 1 ], name = "to_write_scores_transpose" )

    #     #     # reshape scores
    #     #     b_expand_scores = tf.tile( bin_socres, [ 1, encode_size ] )
    #     #     b_new_scores = tf.reshape( b_expand_scores, [ batch_size, encode_size, self.mem_size ] )
    #     #     b_to_write_scores =  1.0 - tf.transpose( b_new_scores, perm = [ 0, 2, 1 ] )
    #     #     b_to_write_scores_r = tf.reduce_max( b_to_write_scores, axis = 0 )
    #     #     b_to_write_scores_r = tf.expand_dims( b_to_write_scores_r, axis = 2 )

    #     #     # apply noise in features to avoid inicialization bias  
    #     #     to_write_noise = to_write + ( self.memory_lr * noise( to_write ) )

    #     #     # filter features
    #     #     to_write_s = to_write_scores * to_write_noise

    #     #     # reduce batch
    #     #     reduced = tf.expand_dims( tf.reduce_mean( to_write_s, axis = 0 ), axis = 2 )
            
    #     #     # get a % of features
    #     #     new_m = 0.01 * reduced

    #     #     # keep % of old memory and unsed scores
    #     #     old_m = 0.99 * ( ( 1.0 - b_to_write_scores_r ) * self.mem )
            
    #     #     # apply new memory
    #     #     wm = ( ( b_to_write_scores_r * self.mem ) + old_m ) + new_m 
    #     #     op_write = tf.assign( self.mem, wm )

    #     # if self.summary:        

    #     #     tf.summary.scalar( 'memory_diff', tf.reduce_mean( tf.abs( self.mem - wm ) ), family = 'ddqn' )

    #     #     tf.summary.image( '3_to_write', tf.expand_dims( to_write, axis = 3 ), max_outputs = 1, family = 'ddqn' )
    #     #     tf.summary.image( '4_to_write_X_scores', tf.expand_dims( to_write_s, axis = 3 ), max_outputs = 1, family = 'ddqn' )

    #     #     tf.summary.image( '2_inverse_scores', tf.expand_dims( b_to_write_scores, axis = 3 ), max_outputs = 1, family = 'ddqn' )        
    #     #     tf.summary.image( '1_scores', tf.expand_dims( to_write_scores, axis = 3 ), max_outputs = 1, family = 'ddqn' )

    #     # return op_write
    #     return tf.no_op()