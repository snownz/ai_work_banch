import sys
sys.path.append('../../')

import tensorflow as tf
import numpy as np
from random import random

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY

from ann_utils.fully_layer import FullyLayer as fc
from ann_utils.fully_layer import mlp

from ann_utils.models.pre_models import MARIO_CONV, MARIO_DCONV,\
                                        MARIO_ACT_REWARD_ENCODE, MARIO_ACT_REWARD_DECODE,\
                                        MARIO_SIMPLE_GS_CONV, MARIO_SIMPLE_GS_DCONV,\
                                        MARIO_256_CONV, MARIO_256_DCONV

from ann_utils.models.transformer import Transformer, GPT2
from ann_utils.self_attention import Self_Attention_Multi_Head_3D_GB as attn_3d
from ann_utils.models.vae import Simple_Image_Vae
from ann_utils.models.rnd import Intrinsic_Curiosity_Module
from ann_utils.models.reinforcemnt_learning_polices_loss import AC, Qlearning
from ann_utils.models.reinforcemnt_learning_networks import LSTM_Policy

from ann_utils.conv_layer import Conv2DLayer as conv
from ann_utils.models.squeeze_n_excitation import SqueezeNExcitation as senet

from ann_utils.helper import gelu, prelu, softmax, sigmoid, relu,\
                             mse_loss, to_float, noise, categorical_sample ,\
                             l2, flatten, normalized_columns_initializer, norm,\
                             group_regularization, bn

class Transformer_Curiosity_AC(object):

    def __init__(self, action_size, state_size, context_size, t_layers, t_heads ):

        self.act_size = action_size

        vae_c_layers = [
            { 'type': 'conv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'conv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'conv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'conv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
        ]

        vae_d_layers = [
            { 'type': 'dconv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'dconv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'dconv', 'out': 16, 'k': 3, 's': 2, 'dp': 0.25, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'dconv', 'out': 3, 'k': 3, 's': 2, 'dp': 0.0, 'bn': True, 'act': tf.nn.sigmoid, 'bias': True },
        ]

        curiosity_layers = [
            { 'o_act': None, 'h_act': prelu, 'dp': 0.0, 'std': 0.1, 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': prelu, 'dp': 0.0, 'std': 0.1, 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': prelu, 'dp': 0.0, 'std': 0.1, 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': prelu, 'dp': 0.0, 'std': 0.1, 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.vision = Simple_Image_Vae( vae_c_layers, vae_d_layers, ( 1024, state_size ), 3, 0.25, prelu )
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .25, prelu, None )
        self.decision = Transformer( context_size, action_size, prelu, t_layers, t_heads )
        self.critic = fc( 1, "value" )
        self.actor = fc( state_size, "logits" )

        self.policy = AC()

    def __call__( self, s, s_, ca, actions, er, timing, 
                 sequence_size, curiosity_rate, name, mode, samples,
                 global_step, update_curiosity_ever, update_vision_ever,
                 is_training=False, summary=False):
        
        with tf.variable_scope('transformer_dqn_worker_{}'.format( name ) ):
            
            mask = tf.one_hot( timing, sequence_size )
            
            with tf.variable_scope('vision'):
                
                # split time_series
                xs = tf.split( s, sequence_size, axis = 1 )
                xs_ = tf.split( s_, sequence_size, axis = 1 )

                rec_loss = lat_loss = es = ds = rec_loss = es_ = None
                states = []
                for s in range( sequence_size ):

                    encoded_state, decoded_state,\
                    vae_recosntruct_loss, vae_lattent_loss,\
                    vae_encoder_vars, vae_decoder_vars,\
                    encoder_update_ops, decoder_update_ops = \
                    self.vision( tf.squeeze( xs[ s ], axis = 1 ), samples = samples, is_training = is_training, summary = ( s == 0 and summary ) )

                    encoded_state_ = \
                    self.vision( tf.squeeze( xs_[ s ], axis = 1 ), samples = 1, is_training = False, summary = False )

                    # sum all looses from vae from all sequence
                    rec_loss = vae_recosntruct_loss if rec_loss is None else rec_loss + vae_recosntruct_loss
                    lat_loss = vae_lattent_loss if lat_loss is None else lat_loss + vae_lattent_loss

                    # keep value from correct state using mask
                    es = mask[ :, s ][:,tf.newaxis] * encoded_state if es is None else es + mask[ :, s ][:,tf.newaxis] * encoded_state
                    es_ = mask[ :, s ][:,tf.newaxis] * encoded_state_ if es_ is None else es_ + mask[ :, s ][:,tf.newaxis] * encoded_state_
                    
                    # control value
                    ds = tf.reshape( mask[ :, s ][:,tf.newaxis] * flatten( decoded_state ), tf.shape(decoded_state) )  if ds is None\
                        else ds + tf.reshape( mask[ :, s ][:,tf.newaxis] * flatten( decoded_state ), tf.shape(decoded_state) )

                    # accumulate states
                    states.append( encoded_state )
            
            tf.summary.image( family = 'vae_values', name = 'sum_image', tensor = ds, max_outputs = 1 )

            with tf.variable_scope('ac'):
                
                # stack all states and all Actions
                pred_inx = tf.concat( [ tf.expand_dims( x, axis = 1 ) for x in states ], axis = 1 )
                pred_iny = tf.one_hot( actions, self.act_size )

                pred = self.decision( pred_inx, pred_iny, is_training = is_training, summary = summary )

                vf = tf.reshape( self.critic( pred, is_training ), [-1])
                logits = self.actor( pred, is_training )
                sample = categorical_sample( logits, self.act_size )
                probs = softmax( logits )

                dqn_vars = [ w for w in tf.compat.v1.trainable_variables() if '/ac/' in w.name ]

            with tf.variable_scope('curiosity'):
                
                ir_, ir, icm_ir, _,\
                p_loss_, p_loss, icm_loss, inverse_loss,\
                pred_vars_, pred_vars, icm_vars, inverse_vars =\
                self.curiosity( es, es_, ca, self.act_size, is_training = is_training, summary = summary )

            with tf.variable_scope('policy'):

                total_reward = tf.reduce_sum( mask * er, axis = 1 ) + ( curiosity_rate * ( ir * ( ir_ + icm_ir ) ) )
                dqn_loss = self.policy( logits, vf, ca, total_reward, summary = summary )

            # filter vars
            vae_encoder_vars = [ v for v in vae_encoder_vars if name in '/{}/'.format( v.name ) ]
            vae_decoder_vars = [ v for v in vae_decoder_vars if name in '/{}/'.format( v.name ) ]
            dqn_vars         = [ v for v in dqn_vars         if name in '/{}/'.format( v.name ) ]
            pred_vars_       = [ v for v in pred_vars_       if name in '/{}/'.format( v.name ) ]
            pred_vars        = [ v for v in pred_vars        if name in '/{}/'.format( v.name ) ]
            icm_vars         = [ v for v in icm_vars         if name in '/{}/'.format( v.name ) ]
            inverse_vars     = [ v for v in inverse_vars     if name in '/{}/'.format( v.name ) ]

            # losses
            p_loss = tf.reduce_mean( p_loss )
            p_loss_ = tf.reduce_mean( p_loss_ )
            icm_loss = tf.reduce_mean( icm_loss )
            inverse_loss = tf.reduce_mean( inverse_loss )
            rec_loss = tf.reduce_mean( rec_loss )
            lat_loss = tf.reduce_mean( lat_loss )

            all_vars = {
                'vae_encoder_vars': vae_encoder_vars,
                'dqn_vars': dqn_vars,
                'vae_decoder_vars': vae_decoder_vars,
                'pred_vars_': pred_vars_,
                'pred_vars': pred_vars,
                'icm_vars': icm_vars,
                'inverse_vars': inverse_vars
            }

            if summary:
                pass
            
            if mode == -1:
                return all_vars

            if mode == 0:
            
                with tf.variable_scope('gradients'):
                    
                    with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                        # vars = vae_encoder_vars + vae_decoder_vars + pred_vars_ + pred_vars + icm_vars + inverse_vars + dqn_vars
                        vars = vae_encoder_vars + vae_decoder_vars
                        reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                        
                        # total_loss = rec_loss + lat_loss + p_loss_ + p_loss + icm_loss + inverse_loss + dqn_loss + reg
                        total_loss = rec_loss + lat_loss + reg

                        if summary:
                            tf.summary.scalar( name = 'total_loss', tensor = total_loss )
                            tf.summary.scalar( name = 'total_l2', tensor = reg )

                        grads = tf.gradients( total_loss, vars )
                        grads = [ tf.clip_by_value( grad, -1, 1 ) for grad in grads if grad != None ]  
                        
                    return sample, probs, grads, all_vars, [ 'vae_encoder_vars', 'vae_decoder_vars' ]
                                   
            if mode == 1:
            
                with tf.variable_scope('gradients'):
                    
                    # encoder
                    with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                        vars = vae_encoder_vars + vae_decoder_vars
                        reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                        
                        total_loss = rec_loss + lat_loss

                        if summary:
                            tf.summary.scalar( name = 'vision_loss', tensor = total_loss )
                            tf.summary.scalar( name = 'vision_l2', tensor = reg )

                        grads = tf.gradients( total_loss, vars ) * ( global_step % update_vision_ever )
                        vision_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]
                                            
                    # curiosity
                    vars = pred_vars_ + pred_vars + icm_vars + inverse_vars
                    reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                    
                    total_loss = p_loss_ + p_loss + icm_loss + inverse_loss + reg

                    if summary:
                        tf.summary.scalar( name = 'curiosity_loss', tensor = total_loss )
                        tf.summary.scalar( name = 'curiosity_l2', tensor = reg )

                    grads = tf.gradients( total_loss, vars ) * ( global_step % update_curiosity_ever )
                    curiosity_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]
                    
                    # predctor
                    vars = dqn_vars
                    reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                    
                    total_loss = dqn_loss + reg

                    if summary:
                        tf.summary.scalar( name = 'predctor_loss', tensor = total_loss )
                        tf.summary.scalar( name = 'predctor_l2', tensor = reg )

                    grads = tf.gradients( total_loss, vars )
                    predictor_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]

                    return sample, probs, ( vision_grads_and_vars + curiosity_grads_and_vars + predictor_grads_and_vars ), all_vars,\
                         [ 'vae_encoder_vars', 'vae_decoder_vars', 'pred_vars_', 'pred_vars', 'icm_vars', 'inverse_vars', 'dqn_vars' ]
                    
            if mode == 2:
            
                with tf.variable_scope('gradients'):
                    
                    # encoder
                    with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                        vars = vae_encoder_vars + vae_decoder_vars
                        reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                        
                        total_loss = rec_loss + lat_loss + dqn_loss + reg

                        if summary:
                            tf.summary.scalar( name = 'vision_loss', tensor = total_loss )
                            tf.summary.scalar( name = 'vision_l2', tensor = reg )

                        grads = tf.gradients( total_loss, vars ) * ( global_step % update_vision_ever )
                        vision_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]
                                            
                    # curiosity
                    vars = pred_vars_ + pred_vars + inverse_vars
                    reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                    
                    total_loss = p_loss_ + p_loss + icm_loss + inverse_loss + reg

                    if summary:
                        tf.summary.scalar( name = 'curiosity_loss', tensor = total_loss )
                        tf.summary.scalar( name = 'curiosity_l2', tensor = reg )

                    grads = tf.gradients( total_loss, vars ) * ( global_step % update_curiosity_ever )
                    curiosity_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]
                    
                    # icm
                    vars = vae_encoder_vars + icm_vars + inverse_vars
                    reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                    
                    total_loss = icm_loss + inverse_loss + lat_loss + reg

                    if summary:
                        tf.summary.scalar( name = 'icm_loss', tensor = total_loss )
                        tf.summary.scalar( name = 'icm_l2', tensor = reg )

                    grads = tf.gradients( total_loss, vars ) * ( global_step % update_curiosity_ever )
                    icm_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]
                    
                    # predctor
                    vars = dqn_vars
                    reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                    
                    total_loss = dqn_loss + reg

                    if summary:
                        tf.summary.scalar( name = 'predctor_loss', tensor = total_loss )
                        tf.summary.scalar( name = 'predctor_l2', tensor = reg )

                    grads = tf.gradients( total_loss, vars )
                    predictor_grads_and_vars = [ tf.clip_by_value( grad, -10, 10 ) for grad, in grads if grad != None ]
                                                            
                    return sample, probs, ( vision_grads_and_vars + curiosity_grads_and_vars + icm_grads_and_vars + predictor_grads_and_vars ), all_vars,\
                           [ 'vae_encoder_vars', 'vae_decoder_vars', 'pred_vars_', 'pred_vars', 'inverse_vars', 'vae_encoder_vars', 'icm_vars', 'inverse_vars', 'dqn_vars' ]

class GPT2_Curiosity_AC(object):

    def __init__(self, action_size, state_size, context_size, t_layers, t_heads ):

        self.act_size = action_size

        vae_c_layers = [
            { 'type': 'conv', 'out': 4, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 4, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 8, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 16, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 32, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu }
        ]

        vae_d_layers = [
            { 'type': 'dconv', 'out': 32, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 16, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 8, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 4, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 3, 'k': 4, 's': 2, 'dp': .0, 'bn': False, 'act': None, 'bias': True },
        ]

        curiosity_layers = [
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.vision = Simple_Image_Vae( vae_c_layers, vae_d_layers, ( 1024, state_size ), 3, .0, tf.nn.elu )
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .0, tf.nn.elu, None )
        self.decision = GPT2( n_ctx = 1024, n_head = 6, n_layer = 6 )
        self.inverse = fc( state_size, "inverse" )
        self.critic = fc( 1, "value" )
        self.actor = fc( action_size, "logits" )

        self.policy = AC()

    def __call__( self, s, s_, past, ac, er, timing, 
                 sequence_size, curiosity_rate, name, mode, samples,
                 global_step, update_curiosity_ever, update_vision_ever, update_predctor_ever,
                 is_training=False, summary=False):
        
        with tf.variable_scope('transformer_dqn_worker_{}'.format( name ) ):
            
            mask = tf.one_hot( timing, sequence_size )
            
            with tf.variable_scope('vision'):
                
                # split time_series
                xs = tf.split( s, sequence_size, axis = 1 )
                xs_ = tf.split( s_, sequence_size, axis = 1 )

                rec_loss = lat_loss = es = ds = dsa = rec_loss = es_ = None
                states = []
                decodeds = []
                for s in range( sequence_size ):

                    msk = mask[ :, s ][:,tf.newaxis]

                    encoded_state, decoded_state,\
                    vae_recosntruct_loss, vae_lattent_loss,\
                    vae_encoder_vars, vae_decoder_vars,\
                    encoder_update_ops, decoder_update_ops = \
                    self.vision( tf.squeeze( xs[ s ], axis = 1 ), samples = samples, is_training = is_training, summary = ( s == 0 and summary ) )

                    encoded_state_ = \
                    self.vision( tf.squeeze( xs_[ s ], axis = 1 ), samples = 1, is_training = False, summary = False )

                    # sum all looses from vae from all sequence
                    rec_loss = vae_recosntruct_loss if rec_loss is None else rec_loss + vae_recosntruct_loss
                    lat_loss = vae_lattent_loss if lat_loss is None else lat_loss + vae_lattent_loss

                    # keep value from correct state using mask
                    es = msk * encoded_state if es is None else es + msk * encoded_state
                    es_ = msk * encoded_state_ if es_ is None else es_ + msk * encoded_state_
                    
                    # control value
                    ds = tf.reshape( msk * flatten( decoded_state ), tf.shape( decoded_state ) )  if ds is None\
                        else ds + tf.reshape( msk * flatten( decoded_state ), tf.shape( decoded_state ) )

                    dsa = decoded_state if dsa is None else dsa + decoded_state

                    # accumulate states
                    states.append( encoded_state )
                    decodeds.append( decoded_state )

            rec_loss /= sequence_size
            lat_loss /= sequence_size

            with tf.variable_scope('ac'):
                
                # stack all states and all Actions
                pred_inx = tf.concat( [ tf.expand_dims( x, axis = 1 ) for x in states ], axis = 1 )

                pred, ctx = self.decision( pred_inx, past, is_training = is_training, summary = summary )

                vf = tf.reshape( self.critic( pred, is_training ), [-1])
                logits = tf.nn.relu( self.actor( pred, is_training ) )
                sample = categorical_sample( logits, self.act_size )
                probs = softmax( logits )

                iv = self.inverse( pred, is_training )
                
                ac_vars = [ w for w in tf.compat.v1.trainable_variables() if '/ac/' in w.name ]

            with tf.variable_scope('curiosity'):
                
                ir_, ir, icm_ir, _,\
                p_loss_, p_loss, icm_loss, inverse_loss,\
                pred_vars_, pred_vars, icm_vars, inverse_vars =\
                self.curiosity( es, es_, ac, self.act_size, is_training = is_training, summary = summary )

            # filter vars
            vae_encoder_vars = [ v for v in vae_encoder_vars if name in '/{}/'.format( v.name ) ]
            vae_decoder_vars = [ v for v in vae_decoder_vars if name in '/{}/'.format( v.name ) ]
            ac_vars          = [ v for v in ac_vars          if name in '/{}/'.format( v.name ) ]
            pred_vars_       = [ v for v in pred_vars_       if name in '/{}/'.format( v.name ) ]
            pred_vars        = [ v for v in pred_vars        if name in '/{}/'.format( v.name ) ]
            icm_vars         = [ v for v in icm_vars         if name in '/{}/'.format( v.name ) ]
            inverse_vars     = [ v for v in inverse_vars     if name in '/{}/'.format( v.name ) ]

            all_vars = {
                'vae_encoder_vars': vae_encoder_vars,
                'ac_vars': ac_vars,
                'vae_decoder_vars': vae_decoder_vars,
                'pred_vars_': pred_vars_,
                'pred_vars': pred_vars,
                'icm_vars': icm_vars,
                'inverse_vars': inverse_vars
            }                
            
            if summary:

                for i, d in enumerate( decodeds ):
                    tf.summary.image( family = 'vision', name = '0_decoded_a_{}'.format(i), tensor = tf.squeeze( xs[ i ], axis = 1 ), max_outputs = 1 )
                    tf.summary.image( family = 'vision', name = '1_decoded_b_{}'.format(i), tensor = d, max_outputs = 1 )

                tf.summary.image( family = 'vision', name = '2_masked_sum_image', tensor = ds, max_outputs = 1 )
                tf.summary.image( family = 'vision', name = '3_sum_image', tensor = dsa, max_outputs = 1 )

                actions = tf.tile( tf.nn.softmax( mask ), [ 1, sequence_size ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, sequence_size, sequence_size ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ sequence_size, sequence_size ] )
                tf.summary.image( family = 'vision', name = '4_sequemce', tensor = actions, max_outputs = 1 )

                actions = tf.tile( tf.nn.softmax( es ), [ 1, 64 ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, 64, es.shape[-1] ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ 64, es.shape[-1] ] )
                tf.summary.image( family = 'vision', name = '5_z_space', tensor = actions, max_outputs = 1 )

                actions = tf.tile( probs, [ 1, self.act_size ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, self.act_size, self.act_size ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ self.act_size, self.act_size ] )
                tf.summary.image( 'actions', actions, max_outputs = 1 )

            if mode == -1:
                return all_vars

            with tf.variable_scope('policy'):

                total_reward = er + ( ir * ( ( ir_ + icm_ir ) / 2.0 ) )
                ac_loss = self.policy( logits, vf, ac, total_reward, total_reward, summary = summary )

            inverse1_loss = tf.losses.mean_squared_error( predictions = iv, labels = es_ )
            
            if summary:

                tf.summary.scalar( 'inverse1_loss', inverse1_loss, family = 'ac' )

                tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
                tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( er ) )
                tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
                tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )
                tf.summary.scalar( family = 'reward', name = 'loss_reward', tensor = tf.reduce_mean( total_reward ) )

            # losses
            p_loss = tf.reduce_mean( p_loss )
            p_loss_ = tf.reduce_mean( p_loss_ )
            icm_loss = tf.reduce_mean( icm_loss )
            inverse_loss = tf.reduce_mean( inverse_loss )
            rec_loss = tf.reduce_mean( rec_loss )
            lat_loss = tf.reduce_mean( lat_loss )

            tf.summary.scalar( family = 'steps', name = 'global_step', tensor = global_step )

            update1 = tf.cond( ( global_step % update_vision_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update2 = tf.cond( ( global_step % update_predctor_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update3 = tf.cond( ( global_step % update_curiosity_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )

            tf.summary.scalar( family = 'steps', name = 'update_vision_ever', tensor = update1 )
            tf.summary.scalar( family = 'steps', name = 'update_predctor_ever', tensor = update2 )
            tf.summary.scalar( family = 'steps', name = 'update_curiosity_ever', tensor = update3 )

            if mode == 0:
            
                with tf.variable_scope('gradients'):
                    
                    with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                        vars = vae_encoder_vars + vae_decoder_vars + pred_vars_ + pred_vars + icm_vars + inverse_vars + ac_vars
                        reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                        
                        total_loss1 = rec_loss + lat_loss + ac_loss + inverse1_loss + reg
                        total_loss2 = update3 * ( p_loss_ + p_loss + icm_loss + reg )
                        total_loss3 = update3 * ( inverse_loss + reg )

                        grads1 = tf.gradients( total_loss1, vae_encoder_vars + vae_decoder_vars )
                        grads1 = [ update1 * tf.clip_by_value( grad, -1, 1 ) for grad in grads1 if grad != None ] 

                        grads2 = tf.gradients( total_loss1, ac_vars )
                        grads2 = [ update2 * tf.clip_by_value( grad, -1, 1 ) for grad in grads2 if grad != None ]

                        grads3 = tf.gradients( total_loss2, pred_vars_ + pred_vars + icm_vars )
                        grads3 = [ update3 * tf.clip_by_value( grad, -1, 1 ) for grad in grads3 if grad != None ]

                        grads4 = tf.gradients( total_loss3, inverse_vars )
                        grads4 = [ update3 * tf.clip_by_value( grad, -1, 1 ) for grad in grads4 if grad != None ]

                        grads = grads1 + grads2 + grads3 + grads4
                        
                    return sample, probs, ctx, grads, all_vars, [ 'vae_encoder_vars', 'vae_decoder_vars', 'ac_vars', 'pred_vars_', 'pred_vars', 'icm_vars', 'inverse_vars' ]

class GPT2_Curiosity_DQN(object):

    def __init__(self, action_size, state_size, context_size, t_layers, t_heads ):


        self.act_size = action_size

        vae_c_layers = [
            { 'type': 'conv', 'out': 4, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 4, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 8, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 16, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 32, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu }
        ]

        vae_d_layers = [
            { 'type': 'dconv', 'out': 32, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 16, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 8, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 4, 'k': 4, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 3, 'k': 4, 's': 2, 'dp': .0, 'bn': False, 'act': None, 'bias': True },
        ]

        curiosity_layers = [
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': 0.1, 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.vision = Simple_Image_Vae( vae_c_layers, vae_d_layers, ( 1024, state_size ), 3, .0, tf.nn.elu )
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .0, tf.nn.elu, None )
        self.decision = GPT2( n_ctx = 1024, n_head = 6, n_layer = 6 )
        self.inverse = fc( state_size, "inverse" )
        self.q_value = fc( action_size, "q_value" )

        self.policy = Qlearning()

    def __call__( self, s, s_, q_value_, past, ac, er, timing, 
                 sequence_size, curiosity_rate, name, mode, samples,
                 global_step, update_curiosity_ever, update_vision_ever, update_predctor_ever,
                 is_training=False, summary=False):
        
        with tf.variable_scope('transformer_dqn_{}'.format( name ) ):
            
            mask = tf.one_hot( timing, sequence_size )
            
            with tf.variable_scope('vision'):
                
                # split time_series
                xs = tf.split( s, sequence_size, axis = 1 )
                xs_ = tf.split( s_, sequence_size, axis = 1 )

                rec_loss = lat_loss = es = ds = dsa = rec_loss = es_ = None
                states = []
                decodeds = []
                for s in range( sequence_size ):

                    msk = mask[ :, s ][:,tf.newaxis]

                    encoded_state, decoded_state,\
                    vae_recosntruct_loss, vae_lattent_loss,\
                    vae_encoder_vars, vae_decoder_vars,\
                    encoder_update_ops, decoder_update_ops = \
                    self.vision( tf.squeeze( xs[ s ], axis = 1 ), samples = samples, is_training = is_training, summary = ( s == 0 and summary ) )

                    encoded_state_ = \
                    self.vision( tf.squeeze( xs_[ s ], axis = 1 ), samples = 1, is_training = False, summary = False )

                    # sum all looses from vae from all sequence
                    rec_loss = vae_recosntruct_loss if rec_loss is None else rec_loss + vae_recosntruct_loss
                    lat_loss = vae_lattent_loss if lat_loss is None else lat_loss + vae_lattent_loss

                    # keep value from correct state using mask
                    es = msk * encoded_state if es is None else es + msk * encoded_state
                    es_ = msk * encoded_state_ if es_ is None else es_ + msk * encoded_state_
                    
                    # control value
                    ds = tf.reshape( msk * flatten( decoded_state ), tf.shape( decoded_state ) )  if ds is None\
                        else ds + tf.reshape( msk * flatten( decoded_state ), tf.shape( decoded_state ) )

                    dsa = decoded_state if dsa is None else dsa + decoded_state

                    # accumulate states
                    states.append( encoded_state )
                    decodeds.append( decoded_state )

            rec_loss /= sequence_size
            lat_loss /= sequence_size
                        
            with tf.variable_scope('dqn'):
                
                # stack all states and all Actions
                pred_inx = tf.concat( [ tf.expand_dims( x, axis = 1 ) for x in states ], axis = 1 )

                pred, ctx = self.decision( pred_inx, past, is_training = is_training, summary = summary )

                q_value = self.q_value( pred, is_training )

                iv = self.inverse( pred, is_training )
                
                dqn_vars = [ w for w in tf.compat.v1.trainable_variables() if '/dqn/' in w.name ]

            with tf.variable_scope('curiosity'):
                
                ir_, ir, icm_ir, _,\
                p_loss_, p_loss, icm_loss, inverse_loss,\
                pred_vars_, pred_vars, icm_vars, inverse_vars =\
                self.curiosity( es, es_, ac, self.act_size, is_training = is_training, summary = summary )

            # filter vars
            vae_encoder_vars = [ v for v in vae_encoder_vars if name in '/{}/'.format( v.name ) ]
            vae_decoder_vars = [ v for v in vae_decoder_vars if name in '/{}/'.format( v.name ) ]
            dqn_vars         = [ v for v in dqn_vars         if name in '/{}/'.format( v.name ) ]
            pred_vars_       = [ v for v in pred_vars_       if name in '/{}/'.format( v.name ) ]
            pred_vars        = [ v for v in pred_vars        if name in '/{}/'.format( v.name ) ]
            icm_vars         = [ v for v in icm_vars         if name in '/{}/'.format( v.name ) ]
            inverse_vars     = [ v for v in inverse_vars     if name in '/{}/'.format( v.name ) ]

            all_vars = {
                'vae_encoder_vars': vae_encoder_vars,
                'dqn_vars': dqn_vars,
                'vae_decoder_vars': vae_decoder_vars,
                'pred_vars_': pred_vars_,
                'pred_vars': pred_vars,
                'icm_vars': icm_vars,
                'inverse_vars': inverse_vars
            }                
            
            if summary:

                for i, d in enumerate( decodeds ):
                    tf.summary.image( family = 'vision', name = '0_decoded_a_{}'.format(i), tensor = tf.squeeze( xs[ i ], axis = 1 ), max_outputs = 1 )
                    tf.summary.image( family = 'vision', name = '1_decoded_b_{}'.format(i), tensor = d, max_outputs = 1 )

                tf.summary.image( family = 'vision', name = '2_masked_sum_image', tensor = ds, max_outputs = 1 )
                tf.summary.image( family = 'vision', name = '3_sum_image', tensor = dsa, max_outputs = 1 )

                actions = tf.tile( tf.nn.softmax( mask ), [ 1, 4 ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, 4, 4 ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ 4, 4 ] )
                tf.summary.image( family = 'vision', name = '4_sequemce', tensor = actions, max_outputs = 1 )

                actions = tf.tile( tf.nn.softmax( es ), [ 1, 64 ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, 64, es.shape[-1] ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ 64, es.shape[-1] ] )
                tf.summary.image( family = 'vision', name = '5_z_space', tensor = actions, max_outputs = 1 )

                actions = tf.tile( q_value, [ 1, self.act_size ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, self.act_size, self.act_size ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ self.act_size, self.act_size ] )
                tf.summary.image( 'actions', actions, max_outputs = 1 )

            if mode == -1:
                return q_value, all_vars

            with tf.variable_scope('policy'):

                total_reward = er + ( ir * ( ( ir_ + icm_ir ) / 2.0 ) )
                # total_reward = ( curiosity_rate * ( ir * ( ir_ + icm_ir ) ) )
                dqn_loss = self.policy( total_reward, q_value_, q_value, ac, summary = summary )

            inverse1_loss = tf.losses.mean_squared_error( predictions = iv, labels = es_ )
            
            if summary:

                tf.summary.scalar( 'inverse1_loss', inverse1_loss, family = 'dqn' )

                tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
                tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( er ) )
                tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
                tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )
                tf.summary.scalar( family = 'reward', name = 'loss_reward', tensor = tf.reduce_mean( total_reward ) )

            # losses
            p_loss = tf.reduce_mean( p_loss )
            p_loss_ = tf.reduce_mean( p_loss_ )
            icm_loss = tf.reduce_mean( icm_loss )
            inverse_loss = tf.reduce_mean( inverse_loss )
            rec_loss = tf.reduce_mean( rec_loss )
            lat_loss = tf.reduce_mean( lat_loss )

            tf.summary.scalar( family = 'steps', name = 'global_step', tensor = global_step )

            update1 = tf.cond( ( global_step % update_vision_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update2 = tf.cond( ( global_step % update_predctor_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update3 = tf.cond( ( global_step % update_curiosity_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )

            tf.summary.scalar( family = 'steps', name = 'update_vision_ever', tensor = update1 )
            tf.summary.scalar( family = 'steps', name = 'update_predctor_ever', tensor = update2 )
            tf.summary.scalar( family = 'steps', name = 'update_curiosity_ever', tensor = update3 )

            if mode == 0:
            
                with tf.variable_scope('gradients'):
                    
                    with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                        vars = vae_encoder_vars + vae_decoder_vars + pred_vars_ + pred_vars + icm_vars + inverse_vars + dqn_vars
                        reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), vars )
                        
                        total_loss1 = rec_loss + lat_loss + dqn_loss + inverse1_loss + reg
                        total_loss2 = update3 * ( p_loss_ + p_loss + icm_loss + reg )
                        total_loss3 = update3 * ( inverse_loss + reg )

                        grads1 = tf.gradients( total_loss1, vae_encoder_vars + vae_decoder_vars )
                        grads1 = [ update1 * tf.clip_by_value( grad, -1, 1 ) for grad in grads1 if grad != None ] 

                        grads2 = tf.gradients( total_loss1, dqn_vars )
                        grads2 = [ update2 * tf.clip_by_value( grad, -1, 1 ) for grad in grads2 if grad != None ]

                        grads3 = tf.gradients( total_loss2, pred_vars_ + pred_vars + icm_vars )
                        grads3 = [ update3 * tf.clip_by_value( grad, -1, 1 ) for grad in grads3 if grad != None ]

                        grads4 = tf.gradients( total_loss3, inverse_vars )
                        grads4 = [ update3 * tf.clip_by_value( grad, -1, 1 ) for grad in grads4 if grad != None ]

                        grads = grads1 + grads2 + grads3 + grads4
                        
                    return q_value, ctx, grads, all_vars, [ 'vae_encoder_vars', 'vae_decoder_vars', 'dqn_vars', 'pred_vars_', 'pred_vars', 'icm_vars', 'inverse_vars' ]

class LSTM_Curiosity_AC(object):

    def __init__(self, input_size, action_size, state_size):

        self.act_size = action_size
        self.input_size = input_size

        vae_c_layers = [
            { 'type': 'conv', 'out': 32, 'k': 8, 's': 4, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 64, 'k': 4, 's': 2, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
            { 'type': 'se', 'act': tf.nn.relu }
        ]

        vae_d_layers = [
            { 'type': 'dconv', 'out': 64, 'k': 3, 's': 4, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 32, 'k': 4, 's': 2, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
            { 'type': 'se', 'act': tf.nn.relu },
            { 'type': 'dconv', 'out': 3,  'k': 8, 's': 1, 'dp': .25, 'bn': True, 'act': None, 'bias': False },
        ]

        curiosity_layers = [
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 512, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.vision = Simple_Image_Vae( vae_c_layers, vae_d_layers, state_size, 3, .0, tf.nn.elu )
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .0, tf.nn.elu, None )
        self.decision = LSTM_Policy( state_size )
        self.inverse = fc( state_size, "inverse" )
        self.critic = fc( 1, "value" )
        self.actor = fc( action_size, "logits" )

        self.policy = AC()

    def __call__( self, s, s_, context, ac, er, adv,
                 name, samples, global_step, 
                 update_curiosity_ever, update_vision_ever, update_predctor_ever,
                 is_training=False, summary=False, compute_loss=False, inference_device='', training_device=''):
                    
        with tf.device( inference_device ):

            s = tf.image.resize_images( s, self.input_size ) / 255.0
            s_ = tf.image.resize_images( s_, self.input_size ) / 255.0

            with tf.variable_scope('vision', reuse = tf.AUTO_REUSE):
                
                mu, logvar,\
                zc, dc,\
                decoded,\
                vae_encoder_vars, vae_decoder_vars,\
                encoder_update_ops, decoder_update_ops = \
                self.vision( s, samples = samples, is_training = is_training, summary = False )

                encoded_state_ = self.vision( s_, samples = 1, is_training = False, summary = False )

                encoded_state = tf.reduce_mean( zc, axis  = 0 )
                decoded_state = tf.reduce_mean( dc, axis  = 0 )
                encoded_state_ = tf.reduce_mean( encoded_state_, axis  = 0 )

            with tf.variable_scope('ac'):
                
                pred, state_out = self.decision( encoded_state, is_training = is_training, 
                                                summary = False, state_in = context, name = name )

                vf = tf.reshape( self.critic( pred, is_training ), [-1])
                logits = tf.nn.relu( self.actor( pred, is_training ) )
                sample = categorical_sample( logits, self.act_size )
                probs = softmax( logits )

                iv = self.inverse( pred, is_training )
                
                ac_vars = [ w for w in tf.compat.v1.trainable_variables() if '/ac/' in w.name ]
            
            with tf.variable_scope('curiosity'):
                                
                ir_, ir, icm_ir, inverse,\
                pred_vars_, pred_vars, icm_vars, inverse_vars =\
                self.curiosity( encoded_state, encoded_state_, ac, self.act_size, is_training = is_training, summary = False )

                curiosity = ir * ( ( ir_ + icm_ir ) / 2.0 )

        # filter vars
        vae_encoder_vars = [ v for v in vae_encoder_vars if name in '/{}/'.format( v.name ) ]
        vae_decoder_vars = [ v for v in vae_decoder_vars if name in '/{}/'.format( v.name ) ]
        ac_vars          = [ v for v in ac_vars          if name in '/{}/'.format( v.name ) ]
        pred_vars_       = [ v for v in pred_vars_       if name in '/{}/'.format( v.name ) ]
        pred_vars        = [ v for v in pred_vars        if name in '/{}/'.format( v.name ) ]
        icm_vars         = [ v for v in icm_vars         if name in '/{}/'.format( v.name ) ]
        inverse_vars     = [ v for v in inverse_vars     if name in '/{}/'.format( v.name ) ]

        all_vars = {
            'vae_encoder_vars': vae_encoder_vars,
            'vae_decoder_vars': vae_decoder_vars,
            'ac_vars': ac_vars,
            'pred_vars_': pred_vars_,
            'pred_vars': pred_vars,
            'icm_vars': icm_vars,
            'inverse_vars': inverse_vars            
        }

        if summary:

            for w in vae_encoder_vars:
                tf.summary.histogram( family = 'vae_encoder_vars', name = w.name.replace(':', '_'), values = w )

            for w in vae_decoder_vars:
                tf.summary.histogram( family = 'vae_decoder_vars', name = w.name.replace(':', '_'), values = w )

            for w in ac_vars:
                tf.summary.histogram( family = 'ac_vars', name = w.name.replace(':', '_'), values = w )

            for w in pred_vars_:
                tf.summary.histogram( family = 'pred_vars_', name = w.name.replace(':', '_'), values = w )

            for w in pred_vars:
                tf.summary.histogram( family = 'pred_vars', name = w.name.replace(':', '_'), values = w )

            for w in icm_vars:
                tf.summary.histogram( family = 'icm_vars', name = w.name.replace(':', '_'), values = w )

            for w in inverse_vars:
                tf.summary.histogram( family = 'inverse_vars', name = w.name.replace(':', '_'), values = w )

        
        if not compute_loss:
            return vf, probs, state_out, curiosity, None, all_vars, all_vars.keys()

        with tf.device( training_device ):
            
            with tf.compat.v1.variable_scope('vae_loss'):

                rec_loss = 0
                for recon_s in decoded:
                    exp = ( flatten( s ) - flatten( recon_s ) ) ** 2
                    rec_loss += tf.reduce_mean( exp, axis = 1 )
                rec_loss /= len( decoded )

                lat_loss = -0.5 * tf.reduce_sum( 1.0 + logvar - tf.pow( mu, 2 ) - tf.exp( logvar ), axis = 1 )
                lat_loss /= to_float( logvar.shape[-1] )

                rec_loss = tf.reduce_mean( rec_loss )
                lat_loss = tf.reduce_mean( lat_loss )

            with tf.compat.v1.variable_scope('curiosity_loss'):

                p_loss = tf.reduce_mean( ir )
                p_loss_ = tf.reduce_mean( ir_ )
                icm_loss = tf.reduce_mean( icm_ir )
                inverse_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = inverse, labels = ac ) )
                                
            with tf.variable_scope('policy_loss'):

                ac_loss = self.policy( logits, vf, ac, er, adv, summary = summary )
                inverse1_loss = tf.losses.mean_squared_error( predictions = iv, labels = encoded_state_ )
            
            update1 = tf.cond( ( global_step % update_vision_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update2 = tf.cond( ( global_step % update_predctor_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update3 = tf.cond( ( global_step % update_curiosity_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )

            print(rec_loss)
            print(lat_loss)
            print(p_loss)
            print(p_loss_)
            print(icm_loss)
            print(inverse_loss)
            print(ac_loss)
            print(inverse1_loss)
            print(update1)
            print(update2)
            print(update3)

            if summary:

                tf.summary.scalar( 'inverse1_loss', inverse1_loss, family = 'ac' )

                tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
                tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( er ) )
                tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
                tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )

                tf.summary.image( family = 'vision', name = '1_decoded', tensor = decoded_state, max_outputs = 1 )

                actions = tf.tile( probs, [ 1, self.act_size ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, self.act_size, self.act_size ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ self.act_size, self.act_size ] )
                tf.summary.image( 'actions', actions, max_outputs = 1 )

                tf.summary.scalar( family = 'steps', name = 'global_step', tensor = global_step )

                tf.summary.scalar( family = 'steps', name = 'update_vision_ever', tensor = update1 )
                tf.summary.scalar( family = 'steps', name = 'update_predctor_ever', tensor = update2 )
                tf.summary.scalar( family = 'steps', name = 'update_curiosity_ever', tensor = update3 )
                
            with tf.variable_scope('gradients'):
                
                with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):
                    
                    total_loss1 = update1 * ( rec_loss + lat_loss + tf.contrib.layers.apply_regularization( l2( 10e-4 ), vae_encoder_vars + vae_decoder_vars ) )
                    total_loss2 = update1 * ( ac_loss + inverse1_loss + tf.contrib.layers.apply_regularization( l2( 10e-4 ), vae_encoder_vars ) )
                    total_loss3 = update2 * ( ac_loss + inverse1_loss + tf.contrib.layers.apply_regularization( l2( 10e-4 ), ac_vars ) )
                    total_loss4 = update3 * ( p_loss_ + p_loss + icm_loss + tf.contrib.layers.apply_regularization( l2( 10e-4 ), pred_vars_ + pred_vars + icm_vars ) )
                    total_loss5 = update3 * ( inverse_loss + tf.contrib.layers.apply_regularization( l2( 10e-4 ), inverse_vars ) )

                    grads1 = tf.gradients( total_loss1, vae_encoder_vars + vae_decoder_vars )
                    grads1 = [ update1 * tf.clip_by_value( grad, -1, 1 ) for grad in grads1 if grad != None ]

                    grads11 = tf.gradients( total_loss2, vae_encoder_vars )
                    grads11 = [ update1 * tf.clip_by_value( grad, -1, 1 ) for grad in grads11 if grad != None ]
                    
                    for i, g in enumerate( grads11 ): grads1[i] += g

                    grads2 = tf.gradients( total_loss3, ac_vars )
                    grads2 = [ update2 * tf.clip_by_value( grad, -1, 1 ) for grad in grads2 if grad != None ]

                    grads3 = tf.gradients( total_loss4, pred_vars_ + pred_vars + icm_vars )
                    grads3 = [ update3 * tf.clip_by_value( grad, -1, 1 ) for grad in grads3 if grad != None ]

                    grads4 = tf.gradients( total_loss5, inverse_vars )
                    grads4 = [ update3 * tf.clip_by_value( grad, -1, 1 ) for grad in grads4 if grad != None ]

                    grads = grads1 + grads2 + grads3 + grads4
                    
                return vf, probs, state_out, curiosity, grads, all_vars, [ 'vae_encoder_vars', 'vae_decoder_vars', 'ac_vars', 'pred_vars_', 'pred_vars', 'icm_vars', 'inverse_vars' ]

class LSTM_Curiosity_Memory_AC(object):

    def __init__(self, input_size, action_size, state_size):

        self.act_size = action_size
        self.input_size = input_size

        curiosity_layers = [
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 128, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 128, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.attn = Self_Attention_Multi_Head_3D_GB( 3, 'attn', key_size = 3, heads = 1, act = None, out_act = tf.nn.sigmoid )
        self.vision = Simple_Image_Vae( MARIO_SIMPLE_GS_CONV, MARIO_SIMPLE_GS_DCONV, state_size, 3, .0, tf.nn.elu )
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .0, tf.nn.elu, None )
        self.decision = LSTM_Policy( state_size )
        self.inverse = fc( state_size, "inverse" )
        self.critic = fc( 1, "value", initializer = normalized_columns_initializer(0.01) )
        self.actor = fc( action_size, "logits", initializer = normalized_columns_initializer(1.0)  )

        self.policy = AC()

    def __call__( self, s, s_, context, ac, er, adv,
                 name, samples, global_step, 
                 update_curiosity_ever, update_vision_ever, update_predctor_ever,
                 is_training=False, summary=False, compute_loss=False, inference_device='', training_device=''):
                   
        with tf.device( inference_device ):

            s = tf.image.resize_images( s, self.input_size ) / 255.0
            s_ = tf.image.resize_images( s_, self.input_size ) / 255.0

            with tf.variable_scope('attention'):

                s_attn, attn_vars = self.attn( s, is_training = is_training, summary = summary, reduction = 2 )
                s_attn = norm( s + s_attn, 'norm_attn', is_training = 'is_training' )

                if summary:
                    tf.summary.image( '9_attn', s_attn, max_outputs = 1, family = "self_attention" )
                    tf.summary.histogram( family = 'attn', name = s_attn.name.replace(':', '_'), values = s_attn )

            with tf.variable_scope('vision', reuse = tf.AUTO_REUSE):
                
                mu, logvar,\
                _, dc,\
                decoded,\
                vae_encoder_vars, vae_decoder_vars,\
                encoder_update_ops, decoder_update_ops = \
                self.vision( s, samples = samples, is_training = is_training, summary = False )

                attn_mu, attn_logvar, _, attn_dc, attn_decoded, _, _, _, _ = \
                self.vision( s_attn, samples = samples, is_training = is_training, summary = False )

                encoded_state_ = self.vision( s_, samples = 1, is_training = False, summary = False )

                decoded_state = tf.reduce_mean( dc, axis  = 0 )
                attn_dc = tf.reduce_mean( attn_dc, axis  = 0 )

            # new_info = mario_memory( encoded_state, 512 )
            # encoded_state_new_info = encoded_state + new_info
        
            with tf.variable_scope('ac'):

                one_hot_action = tf.one_hot( ac, self.act_size )

                enc = tf.concat( [ one_hot_action, attn_mu ], axis = 1 )
                
                pred, state_out = self.decision( enc, is_training = is_training, 
                                                summary = False, state_in = context, name = name )

                vf = tf.reshape( self.critic( pred, is_training ), [-1])
                logits = tf.nn.relu( self.actor( pred, is_training ) )
                probs = softmax( logits )

                iv = self.inverse( pred, is_training )
                
                ac_vars = [ w for w in tf.compat.v1.trainable_variables() if '/ac/' in w.name ]
            
            with tf.variable_scope('curiosity'):
                                
                ir_, ir, icm_ir, inverse,\
                pred_vars_, pred_vars, icm_vars, inverse_vars =\
                self.curiosity( mu, encoded_state_, ac, self.act_size, is_training = is_training, summary = False )

                curiosity = ir * ( ( ir_ + icm_ir ) )

        # filter vars
        vae_encoder_vars = [ v for v in vae_encoder_vars if name in '/{}/'.format( v.name ) ]
        vae_decoder_vars = [ v for v in vae_decoder_vars if name in '/{}/'.format( v.name ) ]
        ac_vars          = [ v for v in ac_vars          if name in '/{}/'.format( v.name ) ]
        pred_vars_       = [ v for v in pred_vars_       if name in '/{}/'.format( v.name ) ]
        pred_vars        = [ v for v in pred_vars        if name in '/{}/'.format( v.name ) ]
        icm_vars         = [ v for v in icm_vars         if name in '/{}/'.format( v.name ) ]
        inverse_vars     = [ v for v in inverse_vars     if name in '/{}/'.format( v.name ) ]
        attn_vars        = [ v for v in attn_vars     if name in '/{}/'.format( v.name ) ]

        all_vars = {
            'vae_encoder_vars': vae_encoder_vars,
            'vae_decoder_vars': vae_decoder_vars,
            'ac_vars': ac_vars,
            'pred_vars_': pred_vars_,
            'pred_vars': pred_vars,
            'icm_vars': icm_vars,
            'inverse_vars': inverse_vars,
            'attn_vars': attn_vars,
        }

        if summary:

            for w in vae_encoder_vars:
                tf.summary.histogram( family = 'vae_encoder_vars', name = w.name.replace(':', '_'), values = w )

            for w in vae_decoder_vars:
                tf.summary.histogram( family = 'vae_decoder_vars', name = w.name.replace(':', '_'), values = w )

            for w in ac_vars:
                tf.summary.histogram( family = 'ac_vars', name = w.name.replace(':', '_'), values = w )

            for w in pred_vars_:
                tf.summary.histogram( family = 'pred_vars_', name = w.name.replace(':', '_'), values = w )

            for w in pred_vars:
                tf.summary.histogram( family = 'pred_vars', name = w.name.replace(':', '_'), values = w )

            for w in icm_vars:
                tf.summary.histogram( family = 'icm_vars', name = w.name.replace(':', '_'), values = w )

            for w in inverse_vars:
                tf.summary.histogram( family = 'inverse_vars', name = w.name.replace(':', '_'), values = w )

            for w in attn_vars:
                tf.summary.histogram( family = 'attn_vars', name = w.name.replace(':', '_'), values = w )
        
        if not compute_loss:
            return vf, probs, state_out, curiosity, None, all_vars, all_vars.keys()

        with tf.device( training_device ):
            
            with tf.compat.v1.variable_scope('curiosity_loss'):

                p_loss = tf.reduce_mean( ir )
                p_loss_ = tf.reduce_mean( ir_ )
                icm_loss = tf.reduce_mean( icm_ir )
                inverse_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = inverse, labels = ac ) )

            with tf.compat.v1.variable_scope('vae_loss'):

                rec_loss = 0
                for recon_s in decoded:
                    exp = ( flatten( s ) - flatten( recon_s ) ) ** 2
                    rec_loss += tf.reduce_mean( exp, axis = 1 )
                rec_loss /= len( decoded )

                lat_loss = -0.5 * tf.reduce_mean( 1.0 + logvar - tf.pow( mu, 2 ) - tf.exp( logvar ), axis = 1 )
                lat_loss /= to_float( logvar.shape[-1] )

                rec_loss = tf.reduce_mean( ir_ * rec_loss )
                lat_loss = tf.reduce_mean( ir_ * lat_loss )

                attn_rec_loss = 0
                for recon_s in attn_decoded:
                    exp = ( flatten( s_attn ) - flatten( recon_s ) ) ** 2
                    attn_rec_loss += tf.reduce_mean( exp, axis = 1 )
                attn_rec_loss /= len( attn_decoded )

                attn_lat_loss = -0.5 * tf.reduce_mean( 1.0 + attn_logvar - tf.pow( mu, 2 ) - tf.exp( attn_logvar ), axis = 1 )
                attn_lat_loss /= to_float( attn_logvar.shape[-1] )

                attn_rec_loss = tf.reduce_mean( ir_ * attn_rec_loss )
                attn_lat_loss = tf.reduce_mean( ir_ * attn_lat_loss )

                rec_loss += attn_rec_loss
                lat_loss += attn_lat_loss
                                
            with tf.variable_scope('policy_loss'):

                ac_loss = self.policy( logits, vf, ac, er, adv, summary = summary )
                inverse1_loss = tf.losses.mean_squared_error( predictions = iv, labels = encoded_state_ )
            
            update1 = tf.cond( ( global_step % update_vision_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update2 = tf.cond( ( global_step % update_predctor_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )
            update3 = tf.cond( ( global_step % update_curiosity_ever ) > 0.0, lambda: 0.0, lambda: 1.0 )

            if summary:

                tf.summary.scalar( 'inverse1_loss', inverse1_loss, family = 'ac' )
                tf.summary.scalar( 'rec_loss', rec_loss, family = 'ac' )
                tf.summary.scalar( 'lat_loss', lat_loss, family = 'ac' )

                tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
                tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( er ) )
                tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
                tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )

                tf.summary.image( family = 'vision', name = '0_in', tensor = s, max_outputs = 1 )
                tf.summary.image( family = 'vision', name = '1_decoded', tensor = decoded_state, max_outputs = 1 )

                actions = tf.tile( probs, [ 1, self.act_size ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, self.act_size, self.act_size ] ), axis = 3 )
                actions = tf.image.resize_bicubic( actions, [ self.act_size, self.act_size ] )
                tf.summary.image( 'actions', actions, max_outputs = 1 )

                tf.summary.scalar( family = 'steps', name = 'global_step', tensor = global_step )

                tf.summary.scalar( family = 'steps', name = 'update_vision_ever', tensor = update1 )
                tf.summary.scalar( family = 'steps', name = 'update_predctor_ever', tensor = update2 )
                tf.summary.scalar( family = 'steps', name = 'update_curiosity_ever', tensor = update3 )
                
            with tf.variable_scope('gradients'):
                
                with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                    l2_1 = tf.contrib.layers.apply_regularization( l2( 10e-6 ), [ vl for vl in vae_encoder_vars + vae_decoder_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ] )
                    l2_2 = tf.contrib.layers.apply_regularization( l2( 10e-6 ), [ vl for vl in vae_encoder_vars + attn_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ] )
                    l2_3 = tf.contrib.layers.apply_regularization( l2( 10e-6 ), [ vl for vl in ac_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ] )
                    l2_4 = tf.contrib.layers.apply_regularization( l2( 10e-6 ), [ vl for vl in pred_vars_ + pred_vars + icm_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ] )
                    l2_5 = tf.contrib.layers.apply_regularization( l2( 10e-6 ), [ vl for vl in inverse_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ] )
                    
                    total_loss1 = update1 * ( rec_loss + lat_loss + l2_1 )
                    total_loss2 = update1 * ( ac_loss + inverse1_loss + l2_2 )
                    total_loss3 = update2 * ( ac_loss + inverse1_loss + l2_3 )
                    total_loss4 = update3 * ( p_loss_ + p_loss + icm_loss + l2_4 )
                    total_loss5 = update3 * ( inverse_loss + l2_5 )

                    grads1 = tf.gradients( total_loss1, vae_encoder_vars + vae_decoder_vars )
                    grads1, _ = tf.clip_by_global_norm( grads1, 1.0 )
                    grads1 = [ update1 * grad for grad in grads1 if grad != None ]

                    grads11 = tf.gradients( total_loss2, vae_encoder_vars + attn_vars )
                    grads11, _ = tf.clip_by_global_norm( grads11, 1.0 )
                    grads11 = [ update1 * grad for grad in grads11 if grad != None ]
                    
                    for i, g in enumerate( grads11 ):
                        if i < len( vae_encoder_vars ):
                            grads1[i] += g
                        else:
                            grads1.append( g )

                    grads2 = tf.gradients( total_loss3, ac_vars  )
                    grads2, _ = tf.clip_by_global_norm( grads2, 1.0 )
                    grads2 = [ update2 * grad for grad in grads2 if grad != None ]

                    grads3 = tf.gradients( total_loss4, pred_vars_ + pred_vars + icm_vars )
                    grads3, _ = tf.clip_by_global_norm( grads3, 1.0 )
                    grads3 = [ update3 * grad for grad in grads3 if grad != None ]

                    grads4 = tf.gradients( total_loss5, inverse_vars )
                    grads4, _ = tf.clip_by_global_norm( grads4, 1.0 )
                    grads4 = [ update3 * grad for grad in grads4 if grad != None ]

                    grads = grads1 + grads2 + grads3 + grads4
                    
                return vf, probs, state_out, curiosity, grads, all_vars, [ 'vae_encoder_vars', 'vae_decoder_vars', 'attn_vars', 'ac_vars', 'pred_vars_', 'pred_vars', 'icm_vars', 'inverse_vars' ]

class LSTM_Curiosity_Memory_DQN(object):

    def __init__(self, input_size, action_size, state_size):

        self.act_size = action_size
        self.input_size = input_size

        curiosity_layers = [
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 128, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 128, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.attn = attn_3d( 1, key_size = 1, heads = action_size, act = None, out_act = None )
        
        self.vision = Simple_Image_Vae( MARIO_256_CONV, MARIO_256_DCONV, state_size, .0, tf.nn.elu )
        
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .0, tf.nn.elu, None )
        
        self.ctx = LSTM_Policy( state_size )
        self.inverse = fc( state_size, "inverse" )
        
        self.policy = Qlearning()

    def __call__( self, s, s_, q_, context, ac, a_, er,
                  name, samples, global_step, 
                  update_dm,
                  is_training=False, summary=False, compute_loss=False):
        
        in_state = s

        with tf.variable_scope('environment_understanding'):
        
            with tf.variable_scope('pre_processing'):
                # s = tf.reduce_mean( tf.image.resize_images( s, self.input_size ), axis = 3, keepdims = True ) / 255.0
                s = ( tf.image.resize_images( s, self.input_size ) / 127.5 ) - 1
                si = s
            
            with tf.variable_scope('feature_transform'):
                s = flatten( s )
        
        with tf.variable_scope('action_understanding'):
                    
            with tf.variable_scope('short_memory'):
                pred, state_out = self.ctx( s, is_training = is_training, summary = False, state_in = context, name = name )                
                sm_vars = [ w for w in tf.trainable_variables() if '/short_memory/' in w.name ]

            with tf.variable_scope('decision_making'):
                features = pred
                # features = tf.concat( [ s, pred ], axis = 1 )
                features = tf.nn.dropout( features, 0.25 )
                h = fc( pred.shape[-1] // 2, "dm_h_features", dropout = 0.25, act = relu )( features, is_training = is_training )
                q_value = fc( self.act_size, "dm", dropout = 0.0 )( h, is_training = is_training )
                dm_vars = [ w for w in tf.trainable_variables() if '/decision_making/' in w.name ]

        sm_vars = [ v for v in sm_vars if '{}/'.format( name ) in v.name ]
        dm_vars = [ v for v in dm_vars if '{}/'.format( name ) in v.name ]

        all_vars = {
            'sm_vars': sm_vars,
            'dm_vars': dm_vars
        }

        if not compute_loss:
            return tf.stop_gradient( q_value ), tf.stop_gradient( state_out ), None, None, all_vars, all_vars.keys()
            
        with tf.variable_scope('policy_loss'):
            # pred_index = tf.one_hot( ac, self.act_size )
            dqn_loss = self.policy( er, q_, q_value, ac, summary = summary )
            dqn_loss = tf.reduce_mean( dqn_loss )
        
        with tf.variable_scope('update_rating'):
            update_dm = tf.cond( ( global_step % update_dm ) > 0.0, lambda: 0.0, lambda: 1.0 )
             
        with tf.variable_scope('gradients'):

                with tf.variable_scope('action_understanding_update'):
                
                    with tf.variable_scope('decision'):
                        dm_grads = tf.gradients( dqn_loss, sm_vars + dm_vars )
                        dm_grads = [ tf.clip_by_value( vl, -1.0, 1.0 ) for vl in dm_grads ]

                grads = dm_grads
        
        if summary:

            tf.summary.scalar( 'dqn_loss', dqn_loss, family = 'action_understanding' )
            tf.summary.scalar( 'dqn_reward', tf.reduce_mean( er ), family = 'action_understanding' )
            tf.summary.scalar( 'dqn_reward_min', tf.math.reduce_min( er ), family = 'action_understanding' )
            tf.summary.scalar( 'dqn_reward_max', tf.math.reduce_max( er ), family = 'action_understanding' )
            tf.summary.scalar( 'dqn_reward_std', tf.math.reduce_std( er ), family = 'action_understanding' )
            
            tf.summary.image( family = 'vision', name = 'in_state', tensor = in_state, max_outputs = 1 )
            tf.summary.image( family = 'vision', name = 'v_in_state', tensor = si, max_outputs = 1 )

            actions = tf.tile( q_value, [ 1, self.act_size ] )
            actions = tf.expand_dims( tf.reshape( actions, [ -1, self.act_size, self.act_size ] ), axis = 3 )
            actions = tf.reduce_mean( tf.image.resize_bicubic( actions, [ self.act_size, self.act_size ] ), axis = 0, keepdims = True )
            tf.summary.image( 'actions', actions, max_outputs = 1 )

            tf.summary.scalar( family = 'steps', name = 'global_step', tensor = global_step )
            tf.summary.scalar( family = 'steps', name = 'update_decision_maker', tensor = update_dm )
                                    
        return q_value, state_out, tf.constant( 0.0 ), grads, all_vars, all_vars.keys()

class LSTM_Curiosity_Memory_Imitation(object):

    def __init__(self, input_size, action_size, state_size):

        self.act_size = action_size
        self.input_size = input_size

        curiosity_layers = [
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 128, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': 128, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': action_size, 'hidden': [ 128 ] },
            { 'o_act': None, 'h_act': tf.nn.elu, 'dp': .0, 'std': random(), 'size': state_size, 'hidden': [ 128 ] },
        ]

        self.attn = attn_3d( 1, key_size = 1, heads = action_size, act = None, out_act = None )
        
        self.vision = Simple_Image_Vae( MARIO_256_CONV, MARIO_256_DCONV, state_size, .0, tf.nn.elu )
        
        self.curiosity = Intrinsic_Curiosity_Module( *curiosity_layers, .0, tf.nn.elu, None )
        
        self.ctx = LSTM_Policy( state_size )
        self.inverse = fc( state_size, "inverse" )
        
        self.dm_h_features = []
        self.dm_h_context = []
        self.q_value = []
        for a in range( action_size ):

            dm_h_features = fc( state_size, 
                                "dm_h_features_{}".format( a ) )
            
            dm_h_context = fc( state_size, 
                               "dm_h_context_{}".format( a ) )

            q_value = fc( 1, "q_value_{}".format( a ) )

            self.dm_h_features.append( dm_h_features )
            self.dm_h_context.append( dm_h_context )
            self.q_value.append( q_value )

        self.policy = Qlearning()

    def __call__( self, s, ta, context, name, samples, global_step, sequence_size, 
                  update_curiosity_ever, update_vision_ever, update_predctor_ever,
                  is_training=False, summary=False, compute_loss=False):
                   
        with tf.variable_scope(name):
            
            with tf.variable_scope('pre_processing'):
                for i in range(sequence_size):
                    s[:,i,:,:] = s[:,i,:,:] / 255.0

            with tf.variable_scope('attention'):
                
                latt = []
                for i in range(sequence_size):
                    vs = tf.reshape( s[:,i,:,:], [-1] + s.shape[2:] )
                    attn, _ = self.attn( vs[i], is_training = is_training, summary = False, reduction = 4 )
                    s_attn = bn( s + attn, name = 'norm_attn', is_training = is_training )
                    latt.append( s_attn )

                attn_vars = [ w for w in tf.trainable_variables() if '/attention/' in w.name ]

            with tf.variable_scope('vision', reuse = tf.AUTO_REUSE):
                
                mus = []
                for i in latt:
                    mu, _, _= \
                    self.vision( i, samples = 1, is_training = False, summary = False )
                    mus.append(mu)
             
            with tf.variable_scope( 'context', reuse = tf.AUTO_REUSE ):
                
                preds = [] 
                for m in mus:                    
                    pred, context = self.ctx( mu, is_training = is_training, summary = False, state_in = context, name = name )
                    preds.append( pred )
                
                context_vars = [ w for w in tf.trainable_variables() if '/context/' in w.name ]

            with tf.variable_scope('dqn'):
                
                
                q_values = []
                dqn_vars = []

                xf = mu
                xc = pred if compute_loss else pred

                for i, vl in enumerate( zip( self.dm_h_features, self.dm_h_context, self.q_value ) ):
                    f, c, q = ( vl[0], vl[1], vl[2] )

                    with tf.variable_scope(str(i)):

                        features = f( xf, is_training )
                        context = c( xc, is_training )

                        xq = tf.concat( [ features, context ], axis = 1 )

                        q_v = q( xq, is_training )

                        q_values.append( q_v )

                        q_vars = [ w for w in tf.trainable_variables() if '/dqn/{}/'.format(i) in w.name and '{}/'.format( name ) in w.name ]

                        dqn_vars.append( q_vars )
                
                q_value = tf.concat( q_values, axis = 1 ) 
                            
            attn_vars        = [ v for v in attn_vars        if '{}/'.format( name ) in v.name ]
            vae_encoder_vars = [ v for v in vae_encoder_vars if '{}/'.format( name ) in v.name ]
            vae_decoder_vars = [ v for v in vae_decoder_vars if '{}/'.format( name ) in v.name ]
            context_vars     = [ v for v in context_vars     if '{}/'.format( name ) in v.name ]

            all_vars = {
                'vae_encoder_vars': vae_encoder_vars,
                'vae_decoder_vars': vae_decoder_vars,
                'context_vars': context_vars,
                'attn_vars': attn_vars
            }
            all_vars.update( { 'dqn_{}'.format( i ): v for i, v in enumerate( dqn_vars ) } )

            if not compute_loss:
                return q_value, state_out, None, None, all_vars, all_vars.keys()

            # with tf.variable_scope('curiosity'):
                                
            #     ir_, ir, icm_ir, inverse,\
            #     pred_vars_, pred_vars, icm_vars, inverse_vars =\
            #     self.curiosity( mu, encoded_s_, ac, self.act_size, is_training = is_training, summary = False )

            #     curiosity = ir * ( ( ir_ + icm_ir ) )

            # filter vars        
            # pred_vars_   = [ v for v in pred_vars_   if name in '/{}/'.format( v.name ) ]
            # pred_vars    = [ v for v in pred_vars    if name in '/{}/'.format( v.name ) ]
            # icm_vars     = [ v for v in icm_vars     if name in '/{}/'.format( v.name ) ]
            # inverse_vars = [ v for v in inverse_vars if name in '/{}/'.format( v.name ) ]
            
            # all_vars['pred_vars_'] = pred_vars_
            # all_vars['pred_vars'] = pred_vars
            # all_vars['icm_vars'] = icm_vars
            # all_vars['inverse_vars'] = inverse_vars
                
            # with tf.compat.v1.variable_scope('curiosity_loss'):

            #     p_loss = tf.reduce_mean( ir )
            #     p_loss_ = tf.reduce_mean( ir_ )
            #     icm_loss = tf.reduce_mean( icm_ir )
            #     inverse_action_loss = tf.reduce_mean( 
            #         tf.nn.sparse_softmax_cross_entropy_with_logits( logits = inverse, labels = ac ) )

            with tf.compat.v1.variable_scope('vae_loss'):

                rec_loss = 0
                for recon_s in decoded:
                    exp = ( flatten( s ) - flatten( recon_s ) ) ** 2
                    rec_loss += tf.reduce_mean( exp, axis = 1 )
                rec_loss /= len( decoded )

                lat_loss = -0.5 * tf.reduce_sum( 1.0 + logvar - tf.square( mu ) - tf.exp( logvar ), axis = 1 )
                lat_loss /= to_float( logvar.shape[-1] )

                rec_loss = tf.reduce_mean( rec_loss )
                lat_loss = tf.reduce_mean( lat_loss )

            with tf.variable_scope('context_loss'):

                context_loss_reconstruct = tf.reduce_mean( ( flatten( s_ ) - flatten( inverse_decoded ) ) ** 2 )
                context_loss_predict_next = tf.reduce_mean( tf.sqrt( tf.square( zs_[0] - iv ) ) )
                context_loss = 0.5 * context_loss_reconstruct + context_loss_predict_next
                
            with tf.variable_scope('policy_loss'):
                
                pred_index = tf.one_hot( ac, self.act_size )
                dqn_loss = self.policy( er, q_, q_value, ac, summary = summary )
            
            with tf.variable_scope('update_rating'):

                update1 = tf.stop_gradient( tf.cond( ( global_step % update_vision_ever ) > 0.0, lambda: 0.0, lambda: 1.0 ) )
                update2 = tf.stop_gradient( tf.cond( ( global_step % update_predctor_ever ) > 0.0, lambda: 0.0, lambda: 1.0 ) )
                update3 = tf.stop_gradient( tf.cond( ( global_step % update_curiosity_ever ) > 0.0, lambda: 0.0, lambda: 1.0 ) )
                
            with tf.variable_scope('loss_N_regularizers'):
                
                with tf.variable_scope('vision'):

                    vision_vars = vae_encoder_vars + vae_decoder_vars
                    vision_w_b = [ vl for vl in vision_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ]
                    l2_vision = l2( 10e-4, vision_w_b )
                    vision_loss = update1 * ( rec_loss + lat_loss + l2_vision )

                with tf.variable_scope('context'):

                    context_loss = update1 * context_loss
                
                with tf.variable_scope('decision'):

                    bs = tf.shape( q_value )[0]
                    _dqn_vars = [ [ v for v in d if '_w_' in v.name ] for d in dqn_vars ]
                    
                    # group lasso
                    g_reg = [ group_regularization( v, True ) for v in _dqn_vars ]
                    g_reg = [ 0.2 * tf.reduce_mean( tf.transpose( tf.reshape( tf.tile( v, [ bs ] ), [ 1, bs ] ) ), axis = 1 ) for v in g_reg ]
                    
                    # masked lasso
                    group_l1_dm = [ pred_index[:,i] * v for i, v in enumerate( g_reg ) ]
                    
                    # summarize loss
                    dm_looses = [ r + dqn_loss for i, r in enumerate( group_l1_dm ) ]
                    dm_looses = [ tf.reduce_mean( v ) for v in dm_looses ]

                    dm_looses_total = update2 * tf.reduce_mean( dm_looses )

                    rewards = [ tf.reduce_mean( pred_index[:,i] * er ) for i, _ in enumerate( g_reg ) ]

                # # curiosity                
                # curiosity_icm_loss = update3 * ( p_loss_ + p_loss + icm_loss )
                # curiosity_inverse_loss = update3 * ( inverse_action_loss )

            if summary:

                tf.summary.scalar( 'loss', vision_loss, family = 'vision' )
                tf.summary.scalar( 'l2', l2_vision, family = 'vision' )
                tf.summary.scalar( 'rec_loss', rec_loss, family = 'vision' )
                tf.summary.scalar( 'lat_loss', lat_loss, family = 'vision' )

                tf.summary.scalar( 'context_loss', context_loss, family = 'ls' )

                tf.summary.scalar( 'dm_loss', dm_looses_total, family = 'ls' )

                # tf.summary.scalar( 'curiosity_inverse_loss', curiosity_inverse_loss, family = 'ls' )
                # tf.summary.scalar( 'curiosity_icm_loss', curiosity_icm_loss, family = 'ls' )
                
                for l, lb in zip( dm_looses, COMPLEX_MOVEMENT ):
                    tf.summary.scalar( str(lb), l, family = 'ls' )

                for l, lb in zip( g_reg, COMPLEX_MOVEMENT ):
                    tf.summary.scalar( str(lb), tf.reduce_mean( l ), family = 'rg' )

                for l, lb in zip( rewards, COMPLEX_MOVEMENT ):
                    tf.summary.scalar( str(lb), tf.reduce_mean( l ), family = 'rw' )

                # tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
                # tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( er ) )
                # tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
                # tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )

                tf.summary.image( family = 'vision', name = '1_in_state', tensor = s, max_outputs = 1 )
                tf.summary.image( family = 'vision', name = '2_in_next', tensor = s_, max_outputs = 1 )

                for d in decoded:
                    tf.summary.image( family = 'vision', name = '1_out_state', tensor = d, max_outputs = 1 )
                tf.summary.image( family = 'vision', name = '2_out_next', tensor = inverse_decoded, max_outputs = 1 )

                actions = tf.tile( q_value, [ 1, self.act_size ] )
                actions = tf.expand_dims( tf.reshape( actions, [ -1, self.act_size, self.act_size ] ), axis = 3 )
                actions = tf.reduce_mean( tf.image.resize_bicubic( actions, [ self.act_size, self.act_size ] ), axis = 0, keepdims = True )
                tf.summary.image( 'actions', actions, max_outputs = 1 )

                tf.summary.scalar( family = 'steps', name = 'global_step', tensor = global_step )

                tf.summary.scalar( family = 'steps', name = 'update_vision_ever', tensor = update1 )
                tf.summary.scalar( family = 'steps', name = 'update_predctor_ever', tensor = update2 )
                tf.summary.scalar( family = 'steps', name = 'update_curiosity_ever', tensor = update3 )

            with tf.variable_scope('gradients'):

                with tf.variable_scope('vision'):
                    vision_grads = tf.gradients( vision_loss, vision_vars )
                    # vision_grads = [ tf.clip_by_value( grad, -1, 1 ) for grad in vision_grads if grad != None ]
                    vision_grads, _ = tf.clip_by_global_norm( vision_grads, 40.0 )

                with tf.variable_scope('context'):
                    context_grads = tf.gradients( context_loss, context_vars )
                    # context_grads = [ tf.clip_by_value( grad, -1, 1 ) for grad in context_grads if grad != None ]
                    context_grads, _ = tf.clip_by_global_norm( context_grads, 40.0 )
                
                with tf.variable_scope('decision'):
                    vd = []
                    for v in dqn_vars: vd += v
                    dm_grads = tf.gradients( dm_looses_total, attn_vars + vd )
                    # dm_grads = [ tf.clip_by_value( grad, -1, 1 ) for grad in dm_grads if grad != None ]
                    dm_grads, _ = tf.clip_by_global_norm( dm_grads, 40.0 )

                # with tf.variable_scope('decision'):
                #     dm_grads = []
                #     dm_attn_grads = []
                #     rate = 1.0 / len( dqn_vars )
                #     for v, l in zip( dqn_vars, dm_looses ):
                #         dm_grad = tf.gradients( l, attn_vars + v )
                #         dm_grad, _ = tf.clip_by_global_norm( dm_grad, 40.0 )
                #         # dm_grad = [ tf.clip_by_value( grad, -1, 1 ) for grad in dm_grad if grad != None ]
                #         dm_grads += dm_grad[len(attn_vars):]

                #         # if len(dm_attn_grads) == 0:
                #         #     dm_attn_grads = [ rate * g for g in dm_grad[:len(attn_vars)] ] 
                #         # else:
                #         #     dm_attn_grads = [ g_ + rate * g for g, g_ in zip( dm_grad[:len(attn_vars)], dm_attn_grads ) ] 

                # curiosity_icm_grad = tf.gradients( curiosity_icm_loss, pred_vars_ + pred_vars + icm_vars )
                # curiosity_icm_grad, _ = tf.clip_by_global_norm( curiosity_icm_grad, 40.0 )

                # curiosity_inverse_grad = tf.gradients( curiosity_inverse_loss, inverse_vars )
                # curiosity_inverse_grad, _ = tf.clip_by_global_norm( curiosity_inverse_grad, 40.0 )

                # grads = vision_grads + context_grads + dm_attn_grads + dm_grads + curiosity_icm_grad + curiosity_inverse_grad
                grads = vision_grads + context_grads + dm_grads
                    
                # return q_value, mu, state_out, curiosity, grads, all_vars,\
                #        [ 'vae_encoder_vars', 'vae_decoder_vars' ]
                        # all_vars.keys()
                
                return q_value, state_out, tf.constant( 0.0 ), grads, all_vars, all_vars.keys()

def mario_memory(encoded_value, size):

    with tf.variable_scope( 'mario_memory' ):

        mem = tf.get_variable( "_mem_", 
                            [ size, encoded_value.shape[-1] ],
                            initializer = tf.initializers.random_normal( mean = 0.0, stddev = 1.0 ), 
                            trainable = False )

        tf.summary.image( family = 'memory', name = '0_memory', tensor = mem[tf.newaxis,:,:,tf.newaxis], max_outputs = 1 )

        logists, scores = mario_memory_score( mem, encoded_value, size )
        # new_info, inverse = new_memory_information( mem, encoded_value, logists, size )
        # new_info_reduced = tf.reduce_mean( new_info, axis = 0 )        
        # newm = new_info_reduced + inverse
        # update = ( mem + newm ) / 2.0
        # update_op = tf.assign( mem, update )
        
        # with tf.control_dependencies( [ update ] ):
        retrived = tf.matmul( scores, mem )

        # tf.summary.image( family = 'memory', name = '1_new_info', tensor = new_info_reduced[tf.newaxis,:,:,tf.newaxis], max_outputs = 1 )
        # tf.summary.image( family = 'memory', name = '2_inverse', tensor = inverse[tf.newaxis,:,:,tf.newaxis], max_outputs = 1 )
        # tf.summary.image( family = 'memory', name = '3_newm', tensor = newm[tf.newaxis,:,:,tf.newaxis], max_outputs = 1 )
        # tf.summary.image( family = 'memory', name = '4_new_memory', tensor = update[tf.newaxis,:,:,tf.newaxis], max_outputs = 1 )

    return retrived

def mario_memory_score(mem, x, size):

    bs = tf.shape( x )[ 0 ]
    vec_size = x.shape[1]
    scores = []
    for s in range( size ):
        
        normalize_mem = tf.reshape( tf.nn.l2_normalize( mem[s,:], 0 ), [ vec_size ] )
        normalize_mem = tf.transpose( tf.reshape( tf.tile( normalize_mem, [ bs ] ), [ vec_size, bs ] ) )
        normalize_x = tf.nn.l2_normalize( x, 1 )
        
        cos_similarity = tf.reduce_sum( tf.multiply( normalize_mem, normalize_x ), axis = 1 )

        scores.append( cos_similarity[:,tf.newaxis] )

    logists = tf.concat( scores, axis = 1 )
    score = softmax( logists )

    return logists, score

def new_memory_information(mem, x, logists, size):

    eps = tf.random_normal( shape = tf.shape( logists ), mean = 0.5, stddev = 0.5, dtype = tf.float32 )
    scores = softmax( eps * logists )

    bin_socres = to_float( scores >= tf.reduce_max( scores, axis = 1, keepdims = True ) )

    newm = []
    inverse = []
    for s in range( size ):
        nm = bin_socres[:, s] * x
        newm.append( nm[:,tf.newaxis,:] )
        inverse.append( ( ( 1.0 - tf.reduce_mean( bin_socres, axis = 0 )[ s ] ) * mem[s,:] )[tf.newaxis,:] )
    
    newm = tf.concat( newm, axis = 1 )
    inverse = tf.concat( inverse, axis = 0 )

    return newm, inverse

def localization_network(x, is_training, n_f=6):

    B = tf.shape( x )[0]
    H, W, C = x.shape[1:]

    loc_in = H*W*C
    loc_out = 6

    # identity transform
    theta = np.array([[1., 0, 0], [0, 1., 0]])
    theta = theta.astype('float32')
    theta = theta.flatten()

    W_loc = tf.Variable( tf.zeros( [ loc_in, loc_out ] ), name = 'W_loc' )
    b_loc = tf.Variable( initial_value = theta, name = 'b_loc' )
    
    # tie everything together
    fc_loc = tf.matmul(tf.zeros([B, loc_in]), W_loc) + b_loc

    return fc_loc

def spatial_transformer_network(fmap, theta, out_dims=None):

    """
        Spatial Transformer Network layer implementation as described in [1].
        The layer is composed of 3 elements:
        - localization_net: takes the original image as input and outputs
        the parameters of the affine transformation that should be applied
        to the input image.
        - affine_grid_generator: generates a grid of (x,y) coordinates that
        correspond to a set of points where the input should be sampled
        to produce the transformed output.
        - bilinear_sampler: takes as input the original image and the grid
        and produces the output image using bilinear interpolation.
        Input
        -----
        - fmap: output of the previous layer. Can be input if spatial
        transformer layer is at the beginning of architecture. Should be
        a tensor of shape (B, H, W, C).
        - theta: affine transform tensor of shape (B, 6). Permits cropping,
        translation and isotropic scaling. Initialize to identity matrix.
        It is the output of the localization network.
        Returns
        -------
        - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
        Notes
        -----
        [1]: 'Spatial Transformer Networks', Jaderberg et. al,
            (https://arxiv.org/abs/1506.02025)
    """

    # grab input dimensions
    B = tf.shape( fmap )[0]
    H, W = tf.shape( fmap )[1:2]

    # reshape theta to (B, 2, 3)
    theta = tf.reshape( theta, [ B, 2, 3 ] )

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator( out_H, out_W, theta )
    else:
        batch_grids = affine_grid_generator( H, W, theta )

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler( fmap, x_s, y_s )

    return out_fmap

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """

    with tf.variable_scope('get_pixel_value'):

        batch_size, height, width = tf.shape(x)[:2]

        batch_idx = tf.range( 0, batch_size )
        batch_idx = tf.reshape( batch_idx, ( batch_size, 1, 1 ) )
        b = tf.tile( batch_idx, ( 1, height, width ) )

        indices = tf.stack( [ b, y, x ], 3 )

        return tf.gather_nd( img, indices )

def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """

    with tf.variable_scope('affine_grid_generator'):

        num_batch = tf.shape(theta)[0]

        # create normalized 2D grid
        with tf.variable_scope('normalized_2D_grid'):        
            x = tf.linspace(-1.0, 1.0, width)
            y = tf.linspace(-1.0, 1.0, height)
            x_t, y_t = tf.meshgrid(x, y)

        # flatten
        with tf.variable_scope('flatten'):
            x_t_flat = tf.reshape(x_t, [-1])
            y_t_flat = tf.reshape(y_t, [-1])

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        with tf.variable_scope('reshape_homogeneous_form'):
            ones = tf.ones_like( x_t_flat )
            sampling_grid = tf.stack( [ x_t_flat, y_t_flat, ones ] )

        # repeat grid num_batch times
        with tf.variable_scope('repeat_grid_by_batch'):
            sampling_grid = tf.expand_dims( sampling_grid, axis = 0 )
            sampling_grid = tf.tile( sampling_grid, tf.stack( [ num_batch, 1, 1 ] ) )

        # cast to float32 (required for matmul)
        with tf.variable_scope('to_float'):
            theta = tf.cast( theta, 'float32' )
            sampling_grid = tf.cast( sampling_grid, 'float32' )

        # transform the sampling grid - batch multiply
        with tf.variable_scope('sample_grid'):
            batch_grids = tf.matmul( theta, sampling_grid )
            # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape( batch_grids, [ num_batch, 2, height, width ] )

        return batch_grids

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    
    with tf.variable_scope('bilinear_sampler'):

        H, W = tf.shape(img)[1:2]

        with tf.variable_scope('max_coords'):
            max_y = tf.cast( H - 1, 'int32' )
            max_x = tf.cast( W - 1, 'int32' )
            zero = tf.zeros( [], dtype = 'int32' )

        # rescale x and y to [0, W-1/H-1]
        with tf.variable_scope('rescale_coords'):
            x = tf.cast( x, 'float32' )
            y = tf.cast( y, 'float32' )
            x = 0.5 * ( ( x + 1.0 ) * tf.cast( max_x - 1, 'float32' ) )
            y = 0.5 * ( ( y + 1.0 ) * tf.cast( max_y - 1, 'float32' ) )

        # grab 4 nearest corner points for each (x_i, y_i)
        with tf.variable_scope('grab_nearest_corner'):
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        with tf.variable_scope('clip_boxes'):
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        # recast as float for delta calculation
        with tf.variable_scope('get_pixels_inside_boxes'):
            Ia = get_pixel_value(img, x0, y0)
            Ib = get_pixel_value(img, x0, y1)
            Ic = get_pixel_value(img, x1, y0)
            Id = get_pixel_value(img, x1, y1)

            x0 = tf.cast(x0, 'float32')
            x1 = tf.cast(x1, 'float32')
            y0 = tf.cast(y0, 'float32')
            y1 = tf.cast(y1, 'float32')

        # calculate deltas
        # add dimension for addition
        with tf.variable_scope('compute_deltas'):
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)

        # compute output
        with tf.variable_scope('compute_output'):
            out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


"""
with tf.variable_scope('loss_N_regularizers'):
                
with tf.variable_scope('vision'):

    vision_vars = vae_encoder_vars + vae_decoder_vars
    vision_w_b = [ vl for vl in vision_vars if ( '_w_' in vl.name or '_b_' in vl.name ) ]
    l2_vision = l2( 10e-4, vision_w_b )
    vision_loss = update1 * ( rec_loss + lat_loss + l2_vision )

with tf.variable_scope('context'):

    context_loss = update1 * context_loss

with tf.variable_scope('decision'):

    bs = tf.shape( q_value )[0]
    _dqn_vars = [ [ v for v in d if '_w_' in v.name ] for d in dqn_vars ]
    
    # group lasso
    g_reg = [ group_regularization( v, True ) for v in _dqn_vars ]
    g_reg = [ 0.2 * tf.reduce_mean( tf.transpose( tf.reshape( tf.tile( v, [ bs ] ), [ 1, bs ] ) ), axis = 1 ) for v in g_reg ]
    
    # masked lasso
    group_l1_dm = [ pred_index[:,i] * v for i, v in enumerate( g_reg ) ]
    
    # summarize loss
    dm_looses = [ r + dqn_loss for i, r in enumerate( group_l1_dm ) ]
    dm_looses = [ tf.reduce_mean( v ) for v in dm_looses ]

    dm_looses_total = update2 * tf.reduce_mean( dm_looses )

    rewards = [ tf.reduce_mean( pred_index[:,i] * er ) for i, _ in enumerate( g_reg ) ]

    for l, lb in zip( dm_looses, COMPLEX_MOVEMENT ):
                    tf.summary.scalar( str(lb), l, family = 'ls' )

                for l, lb in zip( g_reg, COMPLEX_MOVEMENT ):
                    tf.summary.scalar( str(lb), tf.reduce_mean( l ), family = 'rg' )

                for l, lb in zip( rewards, COMPLEX_MOVEMENT ):
                    tf.summary.scalar( str(lb), tf.reduce_mean( l ), family = 'rw' )
"""