import sys, random, gc
sys.path.append('../../')

import threading, logging, time
from ann_utils.agents.replay_memory import ReplayMemory, ReplayMemoryLSTM, ReplayMemorySequence, ReplayMemoryGPT, MemoryRollout
from ann_utils.helper import l2, flatten
import tensorflow as tf
import numpy as np
import cv2 as cv
import six.moves.queue as queue
from collections import namedtuple
import scipy.signal

from ann_utils.fully_layer import FullyLayer as fc

Batch = namedtuple( "Batch", [ "si", "a", "adv", "r", "terminal", "features" ] )

class Curiosity_DQN_Agent:

    def __init__(
            self,
            sess,
            encoder_model,
            curiosity_model,
            predictor_model,
            policy,
            epsilon=1.,
            replace_target_iter=300,
            curiosity_step=100,
            memory_size=10000,
            batch_size=128,
            curiosity_factor=0.1
    ):
        self.curiosity_factor = curiosity_factor
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.curiosity_step = curiosity_step

        # total learning step
        self.learn_step_counter = 0

        self.optimizer_memory = ReplayMemory( memory_size, batch_size )

        self.curiosity_model = curiosity_model
        self.predictor_model = predictor_model
        self.encoder_model = encoder_model
        self.policy = policy

        self.sess = sess

    def build_agent_brain(self, state_shape):

        # override state tensors
        self.st = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s"  )
        self.s_ = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s_" )
        self.ac = tf.compat.v1.placeholder( tf.int32,   [ None, ], name = "a"     )  # input Action
        self.er = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "ext_r" )  # extrinsic reward
        self.dn = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "dn"    )  # dones

        # build model variables
        self.encoder_model.build_variables(True)
        self.predictor_model.build_variables(True)
        self.curiosity_model.build_variables(True)

        # build lattent encoder( z )
        ( state, decoded_state,
          vae_recosntruct_loss, vae_lattent_loss, attn_loss,
          vae_encoder_vars, vae_attn_vars, vae_decoder_vars,
          encoder_update_ops, attn_update_ops, decoder_update_ops ) = self.encoder_model.build_nets( self.st, samples = 1, is_training = True, summary = True )

        state_ = self.encoder_model.build_nets( self.s_, samples = 1, is_training = False, summary = False )

        # build ddqn model
        q_value, q_value_, self.target_net_update_op, ddqn_vars = self.predictor_model.build_nets( state, state_, is_training = True, summary = True )

        # build rnd nets
        ( ir_, ir, icm_ir, inverse_pred,
          p_loss_, p_loss, icm_loss, inverse_loss,
          pred_vars_, pred_vars, icm_vars, inverse_vars ) = self.curiosity_model.build_nets( state, state_, self.ac, is_training = True, summary = True )

        # build ddqn loss
        ddqn_loss = self.policy.get_policy_loss(
            ir * ( self.er + self.curiosity_factor * ( ir_ + icm_ir ) ),
            q_value,
            q_value_,
            self.ac,
            True )

        tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
        tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( self.er ) )
        tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
        tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )
        tf.summary.scalar( 'loss_reward', tf.reduce_mean( ir * ( self.er + self.curiosity_factor * ( ir_ + icm_ir ) ) ), family = 'reward' )

        # self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
        #                                                             predictor.forwardloss * constants['FORWARD_LOSS_WT'])
        # compute gradients
        # grads = tf.gradients( loss, local_var )  # backpropag from global network

        # # clip gradients
        # grads, _ = tf.clip_by_value(grads, constants['GRAD_NORM_CLIP'])
        # grads_and_vars = list( zip( grads, global_var ) ) # apply gradients into local network

        # # copy weights from the parameter server to the local model
        # sync_var_list = [ v1.assign( v2 ) for v1, v2 in zip( local_var, global_var ) ]

        # curiosity optmizer
        with tf.control_dependencies( encoder_update_ops ):

            inv_vars = []
            inv_vars.extend( vae_encoder_vars )
            inv_vars.extend( inverse_vars )

            reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), inv_vars )
            total_inverse_loss = inverse_loss + reg

            tf.summary.scalar( name = 'total_inverse_loss', tensor = total_inverse_loss )
            tf.summary.scalar( name = 'total_inverse_l2', tensor = reg )

            inverse_ops = tf.train.AdamOptimizer( self.curiosity_model.initial_learning_rate ).minimize( total_inverse_loss,
                                                                                                   self.curiosity_model.global_step,
                                                                                                   inv_vars )

        self.curiosity_training_op = [
            tf.train.AdamOptimizer( self.curiosity_model.initial_learning_rate ).minimize( p_loss, self.curiosity_model.global_step, pred_vars ),
            tf.train.AdamOptimizer( self.curiosity_model.initial_learning_rate ).minimize( p_loss_, self.curiosity_model.global_step, pred_vars_ ),
            tf.train.AdamOptimizer( self.curiosity_model.initial_learning_rate ).minimize( icm_loss, self.curiosity_model.global_step, icm_vars ),
            inverse_ops
        ]

        # vae optmizer
        update_ops = []
        update_ops.extend( encoder_update_ops )
        update_ops.extend( decoder_update_ops )
        with tf.control_dependencies(update_ops):

            rec_vars = []
            rec_vars.extend( vae_decoder_vars )
            rec_vars.extend( vae_encoder_vars )

            reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), rec_vars )
            total_loss = vae_recosntruct_loss + vae_lattent_loss + reg

            tf.summary.scalar( name = 'vae_loss', tensor = total_loss )
            tf.summary.scalar( name = 'vae_l2', tensor = reg )

            grads = tf.gradients( total_loss, rec_vars )
            grads_and_vars = list( zip( grads, rec_vars ) )
            grads_and_vars = [ ( tf.clip_by_value( grad, -1, 1 ), var )
                                for grad, var in grads_and_vars if grad != None ]

            self.state_opt = tf.train.AdamOptimizer( self.encoder_model.initial_learning_rate ).apply_gradients( grads_and_vars, self.encoder_model.global_step )

        # q-learning optmizer
        update_ops = []
        update_ops.extend( encoder_update_ops )
        update_ops.extend( attn_update_ops )
        with tf.control_dependencies(update_ops):

            model_vars = []
            model_vars.extend( vae_encoder_vars )
            model_vars.extend( vae_attn_vars )
            model_vars.extend( ddqn_vars )

            reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), model_vars )
            # total_loss = ddqn_loss + ddqn_entropy + attn_loss + reg
            total_loss = ddqn_loss + reg

            tf.summary.scalar( name = 'q_learning_loss', tensor = total_loss )
            tf.summary.scalar( name = 'q_learning_l2', tensor = reg )

            grads = tf.gradients( total_loss, model_vars )
            grads_and_vars = list( zip( grads, model_vars ) )
            grads_and_vars = [ ( tf.clip_by_value( grad, -1, 1 ), var )
                                for grad, var in grads_and_vars if grad != None ]

            self.model_training_op = tf.train.AdamOptimizer( self.predictor_model.initial_learning_rate ).apply_gradients( grads_and_vars, self.predictor_model.global_step )

        # setting variables
        self.q_value = q_value

        self.sess.merge_summary()

    def step(self, s, a, r, s_, done):
        self.optimizer_memory.add( s, a, r, s_, done )

    def choose_action(self, observation):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess( self.q_value, { self.st: s } )
            return np.argmax( actions_value ), True
        else:
            return [ random.randint( 0, 6 ) ]

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.target_net_update_op )

        if len( self.optimizer_memory.memory ) > self.batch_size:

            self.learn_step_counter += 1
            states, actions, rewards, next_states, _, dones = self.optimizer_memory.balanced_sample()

            self.sess( [ self.state_opt, self.model_training_op ],
                       {
                           self.st: states,
                           self.ac: actions,
                           self.er: rewards,
                           self.s_: next_states,
                           self.dn: dones
                       },
                       summary = True,
                       step = self.learn_step_counter )

            # delay training in order to stay curious
            if self.learn_step_counter % self.curiosity_step == 0:

                self.sess( self.curiosity_training_op,
                           {
                               self.st: states,
                               self.ac: actions,
                               self.er: rewards,
                               self.s_: next_states,
                               self.dn: dones
                           } )

class Curiosity_DQN_Context_Agent:

    def __init__(
            self,
            sess,
            encoder_model,
            curiosity_model,
            predictor_model,
            policy,
            epsilon=1.,
            replace_target_iter=300,
            curiosity_step=100,
            memory_size=10000,
            batch_size=128,
            curiosity_factor=0.1
    ):
        self.curiosity_factor = curiosity_factor
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.curiosity_step = curiosity_step

        # total learning step
        self.learn_step_counter = 0

        self.optimizer_memory = ReplayMemoryLSTM( memory_size, batch_size )

        self.curiosity_model = curiosity_model
        self.predictor_model = predictor_model
        self.encoder_model = encoder_model
        self.policy = policy

        self.sess = sess

    def build_agent_brain(self, state_shape, action_size, model_device, training_device):

        # override state tensors
        self.st = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s"  )
        self.s_ = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s_" )
        self.ac = tf.compat.v1.placeholder( tf.int32,   [ None, ], name = "a"     )  # input Action
        self.er = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "ext_r" )  # extrinsic reward
        self.dn = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "dn"    )  # dones

        with tf.device( model_device ):

            # build model variables
            self.encoder_model.build_variables(True)
            self.predictor_model.build_variables(True)
            self.curiosity_model.build_variables(True)

            # build lattent encoder( z )
            state, decoded_state,\
            vae_recosntruct_loss, vae_lattent_loss,\
            vae_encoder_vars, vae_attn_vars, vae_decoder_vars,\
            encoder_update_ops, attn_update_ops, decoder_update_ops = self.encoder_model( self.st, samples = 1, is_training = True, summary = True )

            state_ = self.encoder_model( self.s_, samples = 1, is_training = False, summary = False )

            # build AC model
            q_value, state_out, state_init, state_in, q_value_, \
            self.soft_update_op, ddqn_vars = self.predictor_model( state, state_, is_training = True, summary = True )

            # build curiosity nets
            ir_, ir, icm_ir, inverse_pred,\
            p_loss_, p_loss, icm_loss, inverse_loss,\
            pred_vars_, pred_vars, icm_vars, inverse_vars = self.curiosity_model( state, state_, self.ac, action_size, is_training = True, summary = True )

        with tf.device( training_device ):
            # reward
            total_reward = self.er + ( self.curiosity_factor * ( ir * ( ir_ + icm_ir ) ) )

            # build ddqn loss
            ddqn_loss = self.policy( total_reward, q_value, q_value_, self.ac, True )

            tf.summary.scalar( family = 'reward', name = 'ir', tensor = tf.reduce_mean( ir ) )
            tf.summary.scalar( family = 'reward', name = 'er', tensor = tf.reduce_mean( self.er ) )
            tf.summary.scalar( family = 'reward', name = 'ir_', tensor = tf.reduce_mean( ir_ ) )
            tf.summary.scalar( family = 'reward', name = 'icm', tensor = tf.reduce_mean( icm_ir ) )
            tf.summary.scalar( 'loss_reward', tf.reduce_mean( total_reward ), family = 'reward' )

            # curiosity gradients
            with tf.control_dependencies( encoder_update_ops ):

                inv_vars = vae_encoder_vars + inverse_vars
                inv_factor = list( np.repeat( 0.4, len( vae_encoder_vars ) ) ) + \
                            list( np.repeat( 1.0, len( inverse_vars ) ) )

                reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), inv_vars )
                total_inverse_loss = tf.reduce_mean( inverse_loss + vae_lattent_loss ) + reg

                tf.summary.scalar( name = 'total_inverse_loss', tensor = total_inverse_loss )
                tf.summary.scalar( name = 'total_inverse_l2', tensor = reg )

                grads = tf.gradients( total_inverse_loss, inv_vars )
                grads = [ g * f for g, f in zip( inv_factor, grads ) ]
                grads_and_vars = list( zip( grads, inv_vars ) )
                curiosity_grads_and_vars = [ ( tf.clip_by_value( grad, -10, 10 ), var )
                                            for grad, var in grads_and_vars if grad != None ]

                c_vars = pred_vars + pred_vars_+ icm_vars

                reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), c_vars )
                total_curiosity_loss = tf.reduce_mean( p_loss + p_loss_ + icm_loss ) + reg

                tf.summary.scalar( name = 'total_curiosity_loss', tensor = total_curiosity_loss )
                tf.summary.scalar( name = 'total_curiosity_l2', tensor = reg )

                grads = tf.gradients( total_curiosity_loss, c_vars )
                grads_and_vars = list( zip( grads, c_vars ) )
                curiosity_grads_and_vars += [ ( tf.clip_by_value( grad, -10, 10 ), var )
                                                for grad, var in grads_and_vars if grad != None ]

            # vae gradients
            with tf.control_dependencies( encoder_update_ops + decoder_update_ops ):

                rec_vars = vae_encoder_vars + vae_decoder_vars
                rec_factor = list( np.repeat( 0.5, len( vae_encoder_vars ) ) ) + \
                            list( np.repeat( 1.0, len( vae_decoder_vars ) ) )

                reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), rec_vars )
                total_loss = tf.reduce_mean( vae_recosntruct_loss ) + tf.reduce_mean( vae_lattent_loss ) + reg

                tf.summary.scalar( name = 'vae_loss', tensor = total_loss )
                tf.summary.scalar( name = 'vae_l2', tensor = reg )

                grads = tf.gradients( total_loss, rec_vars )
                grads = [ g * f for g, f in zip( rec_factor, grads ) ]
                grads_and_vars = list( zip( grads, rec_vars ) )
                vae_grads_and_vars = [ ( tf.clip_by_value( grad, -10, 10 ), var )
                                    for grad, var in grads_and_vars if grad != None ]

            # q-learning gradients
            with tf.control_dependencies( encoder_update_ops + attn_update_ops ):

                eval_var = vae_attn_vars + vae_encoder_vars + ddqn_vars
                eval_factor = list( np.repeat( 1.0, len( vae_attn_vars ) ) ) + \
                            list( np.repeat( 0.1, len( vae_encoder_vars ) ) ) + \
                            list( np.repeat( 1.0, len( ddqn_vars ) ) )

                reg = tf.contrib.layers.apply_regularization( l2( 1e-6 ), eval_var )
                total_loss = ddqn_loss + reg

                tf.summary.scalar( name = 'dqn_learning_loss', tensor = total_loss )
                tf.summary.scalar( name = 'dqn_learning_l2', tensor = reg )

                grads = tf.gradients( total_loss, eval_var )
                grads = [ g * f for g, f in zip( eval_factor, grads ) ]
                grads_and_vars = list( zip( grads, eval_var ) )
                q_grads_and_vars = [ ( tf.clip_by_value( grad, -10, 10 ), var )
                                    for grad, var in grads_and_vars if grad != None ]

            # optmizers
            self.curiosity_training_op = tf.train.AdamOptimizer( self.curiosity_model.initial_learning_rate )\
                .apply_gradients( curiosity_grads_and_vars, self.curiosity_model.global_step )

            self.state_opt = tf.train.AdamOptimizer( self.encoder_model.initial_learning_rate )\
                .apply_gradients( vae_grads_and_vars, self.encoder_model.global_step )

            self.model_training_op = tf.train.AdamOptimizer( self.predictor_model.initial_learning_rate )\
                .apply_gradients( q_grads_and_vars, self.predictor_model.global_step )

            # setting variables
            self.q_value = q_value
            self.state_out = state_out
            self.state_init = state_init
            self.state_in = state_in

            self.sess.merge_summary()

    def step(self, s, a, r, s_, state, done):
        self.optimizer_memory.add( s, a, r, s_, state, done )

    def choose_action(self, observation, state, size):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]
        actions_value, states = self.sess(
            [ self.q_value, self.state_out ],
            {
                self.st: s,
                self.state_in[0]: state[0],
                self.state_in[1]: state[1],
            } )

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            return np.argmax( actions_value ), states
        else:
            return random.randint( 0, size - 1 ), states

    def learn(self):

        self.learn_step_counter += 1
        states, actions, rewards, next_states, _, t_states, dones = self.optimizer_memory.balanced_sample()
        t_states = np.transpose( t_states, axes = [ 1, 0, 3, 2 ] )
        t_states = t_states.reshape( t_states.shape[0:3] )

        self.sess( [ self.model_training_op ],
                    {
                        self.st: states,
                        self.ac: actions,
                        self.er: rewards,
                        self.s_: next_states,
                        self.state_in[0]: t_states[0],
                        self.state_in[1]: t_states[1],
                        self.dn: dones
                    },
                    summary = True,
                    step = self.learn_step_counter )

        # delay training in order to stay curious
        if self.learn_step_counter % self.curiosity_step == 0:

            self.sess( [ self.state_opt, self.curiosity_training_op ],
                        {
                            self.st: states,
                            self.s_: next_states,
                            self.ac: actions
                        } )

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.soft_update_op )

class Curiosity_AC_Sequence_Agent:

    def __init__(
            self,
            sess,
            model,
            epsilon=1.,
            replace_target_iter=300,
            curiosity_step=100,
            memory_size=10000,
            batch_size=128,
            curiosity_factor=0.1
    ):
        self.curiosity_factor = curiosity_factor
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.curiosity_step = curiosity_step

        # total learning step
        self.learn_step_counter = 0

        self.optimizer_memory = ReplayMemorySequence( memory_size, batch_size )

        self.model = model

        self.sess = sess

    def reset(self, sequence_size, state_size, action_size):

        self.position = 0
        self.loop = 0
        self.c_state  = list( np.repeat( np.zeros( state_size ), sequence_size ).reshape( [ sequence_size, 1 ] + state_size ) )
        self.c_action = list( np.repeat( 0, sequence_size ).reshape( [ sequence_size, 1, 1 ] ) )
        self.c_reward = list( np.repeat( 0, sequence_size ).reshape( [ sequence_size, 1, 1 ] ) )
        self.c_state_ = list( np.repeat( np.zeros( state_size ), sequence_size ).reshape( [ sequence_size, 1 ] + state_size ) )

    def build_agent_brain(self, state_shape, sequence_size, action_size):

        # override state tensors
        self.st = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s"  )
        self.s_ = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s_" )
        self.ac = tf.compat.v1.placeholder( tf.int32, [ None, sequence_size, ], name = "ac" )
        self.er = tf.compat.v1.placeholder( tf.float32, [ None, sequence_size, ], name = "ext_r" )  # extrinsic reward
        self.s_position = tf.compat.v1.placeholder( tf.int32, [ None, ], name = "s_size" )  # extrinsic reward
        self.current_ac = tf.compat.v1.placeholder( tf.int32, [ None, ], name = "current_ac" )

        global_step = tf.compat.v1.get_variable( 'global_step', [],
                                                  initializer = tf.constant_initializer(0),
                                                  trainable = False )

        learning_rate = tf.compat.v1.train.exponential_decay( 1e-3, global_step, 1000, .97, staircase = False )

        print('\n======= Creating Global Model =======\n')
        # create global model
        g_vars = self.model( self.st, self.s_, self.current_ac, self.ac, self.er, self.s_position,
                             sequence_size, 0.0, 'global', -1, 1, None, 0, 0, is_training = True, summary = False )

        print('\n======= Creating Local Worker 0 =======\n')
        # create worker model
        sample, probs, grads, l0_vars, keys =\
        self.model( self.st, self.s_, self.current_ac, self.ac, self.er, self.s_position,
                    sequence_size, self.curiosity_factor, 'local_0', 0, 4,
                    global_step, 300, 100, is_training = True, summary = True )

        print('\n======= Setting Global with Local Worker 0 =======\n')
        # create shared update
        g_vars_list = g_vars.keys()
        self.soft_update_op = [ ]
        for key in g_vars_list:
            for i, t in enumerate( l0_vars[ key ] ):
                self.soft_update_op.append( tf.assign( t, ( 0.0 * t ) + ( ( 1 - 0.0 ) * g_vars[ key ][ i ] ) ) )

        # order global varibles acording to worker
        ordened_vars = []
        for key in keys: ordened_vars += g_vars[ key ]
        grads_and_vars = zip( grads, ordened_vars )

        print('\n======= Creating Optmizer =======\n')
        # create optmizer
        self.update_opt = tf.train.AdamOptimizer( learning_rate )\
            .apply_gradients( grads_and_vars, global_step )

        # setting variables
        self.sample = sample
        self.probs = probs

        self.sess.merge_summary()

    def step(self, s, a, r, s_):

        s_ = s_[np.newaxis, :]

        self.insert_array( self.position, np.array( [ [ r ] ] ), self.c_reward )
        self.insert_array( self.position, s_, self.c_state_ )

        # if self.position != len( self.c_state ) - 1:

        self.optimizer_memory.add( self.c_state,
                                self.c_action,
                                self.c_reward,
                                self.c_state_,
                                self.position )

        self.position += 1
        if self.position / len( self.c_state ) == 1:
            self.position = len( self.c_state ) - 1

    def insert_array(self, pos, element, array):
        if pos == len( array ) - 1:
            for x in range( 1, pos + 1 ):
                array[ x - 1 ] = array[ x ]
        array[ pos ] = element

    def choose_action(self, observation, size):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        self.insert_array( self.position, s, self.c_state )

        inps = np.transpose( np.asarray( self.c_state ), axes = [ 1, 0, 2, 3, 4 ] )
        inpa = np.transpose( np.squeeze( np.asarray( self.c_action ), axis  = 2) )

        actions_value = self.sess( self.sample,
                                  {
                                      self.st: inps,
                                      self.ac: inpa
                                  } )

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            act = np.argmax( actions_value )
        else:
            act = random.randint( 0, size - 1 )

        self.last_ac = act
        if self.position > 0:
            self.insert_array( self.position - 1, np.array( [ [ self.last_ac ] ] ), self.c_action )

        return act

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.soft_update_op )

        if len( self.optimizer_memory.memory ) > self.batch_size:

            self.learn_step_counter += 1
            states, actions, rewards, next_states, pos = self.optimizer_memory.balanced_sample()

            inps = np.squeeze( states , axis = 2 )
            inps_ = np.squeeze( next_states, axis = 2 )

            inpa = np.squeeze( np.squeeze( actions, axis = 2 ), axis = 2 )
            inpr = np.squeeze( np.squeeze( rewards, axis = 2 ), axis = 2 )

            # TODO convert to numpy operation
            acs = np.asarray( [ x[pos[i]] for i,x in enumerate( inpa ) ] ).astype(int)

            self.sess( self.update_opt,
                       {
                           self.st: inps,
                           self.ac: inpa,
                           self.er: inpr,
                           self.s_: inps_,
                           self.s_position: pos,
                           self.current_ac: acs,
                       },
                       summary = True,
                       step = self.learn_step_counter )

class Curiosity_AC_GPT_Agent:

    def __init__(
            self,
            sess,
            model,
            epsilon=1.,
            replace_target_iter=300,
            curiosity_step=100,
            memory_size=10000,
            batch_size=128,
            curiosity_factor=0.1
    ):
        self.curiosity_factor = curiosity_factor
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.curiosity_step = curiosity_step

        # total learning step
        self.learn_step_counter = 0

        self.optimizer_memory = ReplayMemoryGPT( memory_size, batch_size )

        self.model = model

        self.sess = sess

    def reset(self, sequence_size, state_size, action_size):

        self.register_step = 10
        self.register = 0
        self.position = 0
        self.loop = 0
        self.c_state  = list( np.repeat( np.zeros( state_size ), sequence_size ).reshape( [ sequence_size, 1 ] + state_size ) )
        self.c_state_ = list( np.repeat( np.zeros( state_size ), sequence_size ).reshape( [ sequence_size, 1 ] + state_size ) )
        self.saved_past = np.zeros( [ 1, 8, 2, 6 ] + [ sequence_size, 256 ] )

    def build_agent_brain(self, state_shape, sequence_size, action_size):

        # override state tensors
        self.st = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s"  )
        self.s_ = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s_" )
        self.past = tf.compat.v1.placeholder( tf.float32, [ None, 8, 2, 6 ] + [ sequence_size, 256 ], name = "past" )

        self.ac = tf.compat.v1.placeholder( tf.int32, [ None, ], name = "ac" )
        self.er = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "ext_r" )  # extrinsic reward
        self.s_position = tf.compat.v1.placeholder( tf.int32, [ None, ], name = "s_size" )  # extrinsic reward

        global_step = tf.compat.v1.get_variable( 'global_step', [],
                                                  initializer = tf.constant_initializer(0),
                                                  trainable = False )

        learning_rate = tf.compat.v1.train.exponential_decay( 1e-3, global_step, 1000, .97, staircase = False )

        print('\n======= Creating Global Model =======\n')
        # create global model
        g_vars = self.model( self.st, self.s_, self.past, self.ac, self.er, self.s_position,
                             sequence_size, 0.0, 'global', -1, 1, None, 0, 0, 0, is_training = True, summary = False )

        print('\n======= Creating Local Worker 0 =======\n')
        # create worker model
        sample, probs, present, grads, l0_vars, keys =\
        self.model( self.st, self.s_, self.past, self.ac, self.er, self.s_position,
                    sequence_size, self.curiosity_factor, 'local_0', 0, 2,
                    global_step, 300, 1, 100, is_training = True, summary = True )

        print('\n======= Setting Global with Local Worker 0 =======\n')
        # create shared update
        g_vars_list = g_vars.keys()
        self.soft_update_op = [ ]
        for key in g_vars_list:
            for i, t in enumerate( l0_vars[ key ] ):
                self.soft_update_op.append( tf.assign( t, ( 0.0 * t ) + ( ( 1 - 0.0 ) * g_vars[ key ][ i ] ) ) )

        # order global varibles acording to worker
        ordened_vars = []
        for key in keys: ordened_vars += g_vars[ key ]
        grads_and_vars = zip( grads, ordened_vars )

        print('\n======= Creating Optmizer =======\n')
        # create optmizer
        self.update_opt = tf.train.AdamOptimizer( 1e-4 )\
            .apply_gradients( grads_and_vars, global_step )

        # setting variables
        self.sample = sample
        self.probs = probs
        self.present = present

        self.sess.merge_summary()

    def step(self, s, a, r, s_):

        if self.register % self.register_step == 0:

            s_ = s_[np.newaxis, :]

            self.insert_array( self.position, s_, self.c_state_ )

            # if self.position != len( self.c_state ) - 1:

            self.optimizer_memory.add( self.c_state,
                                    a,
                                    r,
                                    self.c_state_,
                                    self.saved_past,
                                    self.position )

            self.position += 1
            if self.position / len( self.c_state ) == 1:
                self.position = len( self.c_state ) - 1

        self.register += 1

    def insert_array(self, pos, element, array):
        if pos == len( array ) - 1:
            for x in range( 1, pos + 1 ):
                array[ x - 1 ] = array[ x ]
        array[ pos ] = element

    def choose_action(self, observation, size):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if self.register % self.register_step == 0:
            self.insert_array( self.position, s, self.c_state )

        inps = np.transpose( np.asarray( self.c_state ), axes = [ 1, 0, 2, 3, 4 ] )

        actions_value, self.saved_past = self.sess( [ self.sample, self.present ],
                                  {
                                      self.st: inps,
                                      self.past: self.saved_past
                                  } )

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            act = np.argmax( actions_value )
        else:
            act = random.randint( 0, size - 1 )

        return act

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.soft_update_op )

        if len( self.optimizer_memory.memory ) > self.batch_size:

            self.learn_step_counter += 1
            states, actions, rewards, next_states, past, pos = self.optimizer_memory.balanced_sample()

            past = np.squeeze( past, axis = 1 )

            inps = np.squeeze( states , axis = 2 )
            inps_ = np.squeeze( next_states, axis = 2 )

            self.sess( self.update_opt,
                       {
                           self.st: inps,
                           self.ac: actions,
                           self.er: rewards,
                           self.s_: inps_,
                           self.s_position: pos,
                           self.past: past
                       },
                       summary = True,
                       step = self.learn_step_counter )

class Curiosity_DQN_GPT_Agent:

    def __init__(
            self,
            sess,
            model,
            epsilon=1.,
            replace_target_iter=300,
            curiosity_step=100,
            memory_size=10000,
            batch_size=128,
            curiosity_factor=0.1
    ):
        self.curiosity_factor = curiosity_factor
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.curiosity_step = curiosity_step

        # total learning step
        self.learn_step_counter = 0

        self.optimizer_memory = ReplayMemoryGPT( memory_size, batch_size )

        self.model = model

        self.sess = sess

    def reset(self, sequence_size, state_size, action_size):

        self.register_step = 3
        self.register = 0
        self.position = 0
        self.loop = 0
        self.c_state  = list( np.repeat( np.zeros( state_size ), sequence_size ).reshape( [ sequence_size, 1 ] + state_size ) )
        self.c_state_ = list( np.repeat( np.zeros( state_size ), sequence_size ).reshape( [ sequence_size, 1 ] + state_size ) )
        self.saved_past = np.zeros( [ 1, 8, 2, 6 ] + [ sequence_size, 256 ] )

    def build_agent_brain(self, state_shape, sequence_size, action_size):

        # override state tensors
        self.st = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s"  )
        self.s_ = tf.compat.v1.placeholder( tf.float32, state_shape, name = "s_" )
        self.past = tf.compat.v1.placeholder( tf.float32, [ None, 8, 2, 6 ] + [ sequence_size, 256 ], name = "past" )

        self.ac = tf.compat.v1.placeholder( tf.int32, [ None, ], name = "ac" )
        self.er = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "ext_r" )  # extrinsic reward
        self.s_position = tf.compat.v1.placeholder( tf.int32, [ None, ], name = "s_size" )  # extrinsic reward

        global_step = tf.compat.v1.get_variable( 'global_step', [],
                                                  initializer = tf.constant_initializer(0),
                                                  trainable = False )

        learning_rate = tf.compat.v1.train.exponential_decay( 1e-3, global_step, 1000, .97, staircase = False )

        print('\n======= Creating Target Model =======\n')
        q_value_, t_vars = self.model( self.s_, self.s_, None, self.past, self.ac, self.er, self.s_position,
                                       sequence_size, 0.0, 'target', -1, 1, None, 0, 0, 0, is_training = True, summary = False )

        print('\n======= Creating Eval Model =======\n')
        q_value, present, grads, e_vars, keys =\
        self.model( self.st, self.s_, q_value_, self.past, self.ac, self.er, self.s_position,
                    sequence_size, self.curiosity_factor, 'eval', 0, 2,
                    global_step, 300, 1, 10, is_training = True, summary = True )

        ordened_vars = []
        for key in keys: ordened_vars += e_vars[ key ]
        grads_and_vars = zip( grads, ordened_vars )

        num_parms = np.sum( np.prod( y.shape ) for y in ordened_vars )

        print('\n======= Setting Target with Eval =======\n')
        # create shared update
        g_vars_list = t_vars.keys()
        self.soft_update_op = [ ]
        for key in g_vars_list:
            for i, e in enumerate( e_vars[ key ] ):
                self.soft_update_op.append( tf.assign( t_vars[ key ][ i ], ( 0.5 * t_vars[ key ][ i ] ) + ( ( 1 - 0.5 ) * e ) ) )

        print('\n======= Creating Optmizer =======\n')
        # create optmizer
        # self.update_opt = tf.train.AdamOptimizer( 1e-3 )\
        # self.update_opt = tf.train.MomentumOptimizer( learning_rate, 0.9 )\
        self.update_opt = tf.train.RMSPropOptimizer( 1e-3 )\
            .apply_gradients( grads_and_vars, global_step )

        # setting variables
        self.q_value = q_value
        self.present = present

        self.sess.merge_summary()

        print('\n======= Params Size: {} =======\n'.format(num_parms))

    def step(self, s, a, r, s_):

        if self.register % self.register_step == 0:

            s_ = s_[np.newaxis, :]

            self.insert_array( self.position, s_, self.c_state_ )

            # if self.position != len( self.c_state ) - 1:

            self.optimizer_memory.add( self.c_state,
                                    a,
                                    r,
                                    self.c_state_,
                                    self.saved_past,
                                    self.position )

            self.position += 1
            if self.position / len( self.c_state ) == 1:
                self.position = len( self.c_state ) - 1

        self.register += 1

    def insert_array(self, pos, element, array):
        if pos == len( array ) - 1:
            for x in range( 1, pos + 1 ):
                array[ x - 1 ] = array[ x ]
        array[ pos ] = element

    def choose_action(self, observation, size):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if self.register % self.register_step == 0:
            self.insert_array( self.position, s, self.c_state )

        inps = np.transpose( np.asarray( self.c_state ), axes = [ 1, 0, 2, 3, 4 ] )

        actions_value, self.saved_past = self.sess( [ self.q_value, self.present ],
                                  {
                                      self.st: inps,
                                      self.past: self.saved_past
                                  } )

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            act = np.argmax( actions_value )
        else:
            act = random.randint( 0, size - 1 )

        return act

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess( self.soft_update_op )

        if len( self.optimizer_memory.memory ) > self.batch_size:

            self.learn_step_counter += 1
            states, actions, rewards, next_states, past, pos = self.optimizer_memory.balanced_sample()

            past = np.squeeze( past, axis = 1 )

            inps = np.squeeze( states , axis = 2 )
            inps_ = np.squeeze( next_states, axis = 2 )

            self.sess( self.update_opt,
                       {
                           self.st: inps,
                           self.ac: actions,
                           self.er: rewards,
                           self.s_: inps_,
                           self.s_position: pos,
                           self.past: past
                       },
                       summary = True,
                       step = self.learn_step_counter )

class Generic(object):
    pass

class Curiosity_AC_Context_Agent:

    def __init__(self, model, memory_size=1000, batch_size=128, n_episodes=100000):
        self.model = model
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.t_steps = memory_size // 2

    def build_agent_brain(
        self, state_shape, state_size, action_size,
        cpu_device, acellerator_device,
        session, render, is_curiosity, noReward, name, compute_grads, vars_sync):

        with tf.variable_scope( name ):

            st  = tf.compat.v1.placeholder( tf.float32, [ None, ] + state_shape, name = "s" )
            s_  = tf.compat.v1.placeholder( tf.float32, [ None, ] + state_shape, name = "s_" )
            ac  = tf.compat.v1.placeholder( tf.int32,   [ None, ], name = "a" )  # input Action
            adv = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "a" )  # input Action
            er  = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "ext_r" )  # extrinsic reward

            # lstm state
            state_in = [ 
                tf.placeholder( tf.float32, [ None, state_size ], name = 'c_in' ), 
                tf.placeholder( tf.float32, [ None, state_size ], name = 'h_in' ) 
            ]
                
            # lstm state
            state_in_ = tf.concat( [ state_in[0], state_in[1] ], axis = 1 )

            # params
            global_step = tf.compat.v1.get_variable( 'global_step', [], initializer = tf.constant_initializer(0), trainable = False )
            
            print('\n======= Creating Model =======\n')
            # create worker model
            critic, probs, state_out, curiosity, grads, vars, keys =\
            self.model( st, s_, state_in_, ac, er, adv,
                        name, 
                        4 if compute_grads else 1,                        
                        global_step, 100, 1, 10, 
                        is_training = True, summary = True, compute_loss = compute_grads,
                        inference_device = cpu_device.name, training_device = acellerator_device.name )
            
            sync_vars = []
            update_opt = []
            summaries_loss = []
            if compute_grads:

                with tf.device( acellerator_device.name ):
                     
                    print('\n======= Sync =======\n')
                    # create shared update
                    for key in keys:
                        for i, t in enumerate( vars[ key ] ):
                            sync_vars.append( tf.assign( t, vars_sync[ key ][ i ] ) )
     
                    # order global varibles acording to worker
                    ordened_vars = []
                    for key in keys: ordened_vars += vars_sync[ key ]
                    grads_and_vars = zip( grads, ordened_vars )

                    print('\n======= Creating Optmizer =======\n')
                    # create optmizer
                    update_opt = tf.train.MomentumOptimizer( 1e-4, 0.99 )\
                        .apply_gradients( grads_and_vars, global_step )
                    
                    summaries_loss = [ x for x in tf.get_collection( tf.GraphKeys.SUMMARIES ) if name in x.name and not 'vars' in x.name ]

                    num_parms = np.sum( np.prod( y.shape ) for y in ordened_vars )

                    print('\n======= Params Size: {} =======\n'.format(num_parms))
            
            summaries = [ x for x in tf.get_collection( tf.GraphKeys.SUMMARIES ) if name in x.name and ( 'vars' in x.name or 'memory' in x.name ) ]

            agent = Generic()
            
            # configs
            agent.name = name
            agent.n_episodes = self.n_episodes
            agent.t_steps = self.t_steps
            agent.action_size = action_size
            agent.render = render
            agent.sess = session
            agent.train = compute_grads
            agent.curiosity = is_curiosity
            agent.noReward = noReward

            # model
            agent.st = st
            agent.s_ = s_
            agent.ac  = ac
            agent.adv = adv
            agent.er  = er
            agent.state_in = state_in
            agent.probs = probs
            agent.bonus = curiosity
            agent.critic = critic
            agent.state_out = state_out
            agent.sync = sync_vars
            agent.update_opt = update_opt
            agent.variables = vars
            agent.summary = tf.summary.merge( summaries )
            agent.global_step = global_step

            if len( summaries_loss ) > 0:
                agent.summary_op = tf.summary.merge( summaries_loss )

            lstm_agent = ACLSTMCuriosityAgent( agent, state_size )

        return lstm_agent

class Curiosity_DDQN_Context_Agent:

    def __init__(self, model, memory_size=1000, batch_size=128, n_episodes=100000):
        self.model = model
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_episodes = n_episodes

    def build(self, state_shape, state_size, action_size, session, is_curiosity, noReward):

        st  = tf.compat.v1.placeholder( tf.float32, [ None, ] + state_shape, name = "s" )
        ac  = tf.compat.v1.placeholder( tf.int32,   [ None, ], name = "a" )  # input Action
        a_  = tf.compat.v1.placeholder( tf.float32, [ None, action_size ], name = "a_" )  # input Action
        er  = tf.compat.v1.placeholder( tf.float32, [ None, ], name = "ext_r" )  # extrinsic reward
        s_ = tf.compat.v1.placeholder( tf.float32, [ None, ] + state_shape, name = "s_" )

        # lstm state
        state_in = [ 
            tf.placeholder( tf.float32, [ None, state_size ], name = 'c_in' ), 
            tf.placeholder( tf.float32, [ None, state_size ], name = 'h_in' ) 
        ]
            
        # lstm state
        state_in_ = tf.concat( [ state_in[0], state_in[1] ], axis = 1 )

        q_next, target_vars, keys =\
        self.build_agent_brain( s_, None, None, state_in_, ac, a_, None, session, False, 'target', False )
        
        q_value, curiosity, state_out, update_opt, eval_vars, global_step =\
        self.build_agent_brain( st, s_, q_next, state_in_, ac, a_, er, session, is_curiosity, 'eval', True )

        with tf.variable_scope('sync_models'):
            sync_vars = []
            print('\n======= Sync =======\n')
            # create shared update
            for key in keys:
                for i, t in enumerate( target_vars[ key ] ):
                    sync_vars.append( tf.assign( t, eval_vars[ key ][ i ] ) )

        summaries_loss = [ x for x in tf.get_collection( tf.GraphKeys.SUMMARIES ) if 'eval' in x.name and not 'vars' in x.name ]
        summaries = [ x for x in tf.get_collection( tf.GraphKeys.SUMMARIES ) if 'eval' in x.name and ( 'vars' in x.name or 'memory' in x.name ) ]

        agent = Generic()
        
        # configs
        agent.name = 'eval'
        agent.n_episodes = self.n_episodes
        agent.action_size = action_size
        agent.sess = session
        agent.train = True
        agent.curiosity = is_curiosity
        agent.noReward = noReward
        agent.render = True
        agent.memory = ReplayMemoryLSTM( self.memory_size, self.batch_size )

        # model
        agent.st = st
        agent.s_ = s_
        agent.ac = ac
        agent.a_ = a_
        agent.er = er
        agent.state_in = state_in
        agent.q_value = q_value
        agent.bonus = curiosity
        agent.state_out = state_out
        agent.sync = sync_vars
        agent.update_opt = update_opt
        agent.variables = eval_vars        
        agent.global_step = global_step

        if len( summaries_loss ) > 0:
            agent.summary_op = tf.summary.merge( summaries_loss )

        if len( summaries ) > 0:
            agent.summary = tf.summary.merge( summaries )
        else:
            agent.summary = None

        return DQNLSTMCuriosityAgent( agent, state_size )

    def build_agent_brain( self, st, s_, q_, state_in_, ac, a_, er,
                           session, is_curiosity, name, compute_grads):

        with tf.variable_scope( name ):

            # params
            global_step = tf.compat.v1.get_variable( 'global_step', [], initializer = tf.constant_initializer(0), trainable = False )
            
            print('\n======= Creating Model =======\n')
            # create worker model
            q_value, state_out, curiosity, grads, vars, keys =\
            self.model( st, s_, q_, state_in_, ac, a_, er,
                        name, 
                        4 if compute_grads else 1,                        
                        global_step, 1, 
                        is_training = True, summary = True, compute_loss = compute_grads )
                        
            if compute_grads:
                
                with tf.variable_scope('updater'):
                    
                    # order global varibles acording to worker
                    ordened_vars = []
                    for key in keys: ordened_vars += vars[ key ]
                    grads_and_vars = zip( grads, ordened_vars )

                    print('\n======= Creating Optmizer =======\n')
                    # create optmizer
                    # update_opt = tf.train.MomentumOptimizer( 1e-4, 0.9 )\
                    update_opt = tf.train.AdamOptimizer( 1e-4, 0.9 )\
                        .apply_gradients( grads_and_vars, global_step )

                    num_parms = np.sum( np.prod( y.shape ) for y in ordened_vars )
                    num_layers = len( [ y for y in ordened_vars if '_w_' in y.name ] )

                    print('\n======= Layers Size: {} =======\n'.format(num_layers))
                    print('\n======= Params Size: {} =======\n'.format(num_parms))

                    return q_value, curiosity, state_out, update_opt, vars, global_step
            
            else:
                return q_value, vars, keys

class A3C:

    def __init__( self, env, envs, model, workers, summary_writer,
                  gamma=0.90, _lambda=1.0, num_local_steps=20, reawrd_clip=1.0 ):
        
        self.explorers = [ ACExplorerThread( ev, w, num_local_steps ) for ev, w in zip( envs, workers ) ]
        self.trainers = [ ACTrainerThread( w, gamma, _lambda, reawrd_clip ) for w in self.explorers ]
        
        self.global_runner = ACExplorerThread( env, model, num_local_steps )
        self.summary_writer = summary_writer
        self.train_op = model.model.update_opt
        self.global_step = model.model.global_step    
        self.gamma = gamma
        self._lambda = _lambda
        self.reawrd_clip = reawrd_clip

    def start(self, w_wiriter, run_global=False):
        if run_global:
            self.global_runner.start_runner( self.summary_writer )
        for e, t, w in zip( self.explorers, self.trainers, w_wiriter ):
            e.start_runner( w )
            t.start_runner( w )

class A2C:

    def __init__( self, env, model, summary_writer,
                  gamma=0.99, _lambda=1.0, num_local_steps=20, reawrd_clip=1.0,
                  curiosity=True, noReward=True ):
        
        model.model.train = False
        self.global_runner = RunnerThread( env, model, num_local_steps, True, True )
        self.summary_writer = summary_writer
        self.summary_op = model.model.summary_op
        self.train_op = model.model.update_opt
        self.global_step = model.model.global_step    
        self.gamma = gamma
        self._lambda = _lambda
        self.reawrd_clip = reawrd_clip

    def start(self):
        self.global_runner.start_runner( self.summary_writer )

    def pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        # get top rollout from queue (FIFO)
        rollout = self.global_runner.queue.get( timeout = 600.0 )
        while not rollout.terminal:
            try:                    
                rollout.extend( self.global_runner.queue.get_nowait() )
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        
        rollout = self.pull_batch_from_queue()
        
        if len( rollout.states ) > 0:

            batch = process_rollout( rollout, gamma = self.gamma, lambda_ = self._lambda, clip = self.reawrd_clip )

            fetches = [ self.summary_op, self.train_op, self.global_step ]

            state = batch.si[:-1]
            state_ = batch.si[1:]
            actions = np.argmax( batch.a.squeeze(1), axis = 1 )
            advantage = batch.adv
            rewards = batch.r
            lstm_c = np.concatenate( [ x[0] for x in batch.features ], axis = 0 )
            lstm_h = np.concatenate( [ x[1] for x in batch.features ], axis = 0 )

            feed_dict = {
                self.global_runner.model.model.st: state,
                self.global_runner.model.model.s_: state_,
                self.global_runner.model.model.ac: actions,
                self.global_runner.model.model.adv: advantage,
                self.global_runner.model.model.er: rewards,
                self.global_runner.model.model.state_in[0]: lstm_c,
                self.global_runner.model.model.state_in[1]: lstm_h,
            }

            fetched = sess( fetches, feed_dict )

            self.summary_writer.add_summary( fetched[0], fetched[-1] )
            self.summary_writer.flush()

class DDQN:

    def __init__(self, env, model, summary_writer, gamma=0.90, num_local_steps=20, reawrd_clip=1.0, update_steps=100):        
        self.explorer = DQNExplorerThread( env, model, num_local_steps, gamma )
        self.trainer = DQNTrainerThread( self.explorer, gamma, reawrd_clip, update_steps )
        self.summary_writer = summary_writer

    def start(self):

        self.explorer.start_runner( self.summary_writer )
        self.trainer.start_runner( self.summary_writer )

class ACLSTMCuriosityAgent():

    def __init__(self, model, state_size):
        self.model = model
        self.state_size = state_size

    def reset(self):
        return np.zeros( [ 1, self.state_size ] ), np.zeros( [ 1, self.state_size ] )

    def predict(self, observation, state, prior_act, summary):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if summary:
            actions, critic, states, sm =\
            self.model.sess( 
                            [ self.model.probs, self.model.critic, self.model.state_out, self.model.summary ],
                            {
                                self.model.st: s,
                                self.model.ac: prior_act,
                                self.model.state_in[0]: state[0],
                                self.model.state_in[1]: state[1]
                            }
                            )

            return actions, critic, states, sm
        else:
            actions, critic, states =\
            self.model.sess( 
                            [ self.model.probs, self.model.critic, self.model.state_out ],
                            {
                                self.model.st: s,
                                self.model.ac: prior_act,
                                self.model.state_in[0]: state[0],
                                self.model.state_in[1]: state[1]
                            }
                            )

            return actions, critic, states, None

    def value(self, observation, state, prior_act):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]
        critic =\
        self.model.sess( 
                         self.model.critic,
                         {
                            self.model.st: s,
                            self.model.ac: prior_act,
                            self.model.state_in[0]: state[0],
                            self.model.state_in[1]: state[1]
                          } 
                        )

        return critic
    
    def get_action(self, actions):
        a = np.random.choice( actions[0], p = actions[0] )
        return np.argmax( actions == a )

    def pred_bonus(self, last_state, state, action):
        
        s = last_state[np.newaxis, :]
        s_ = state[np.newaxis, :]
        
        bonus = self.model.sess( self.model.bonus,
                                 {
                                    self.model.st: s,
                                    self.model.s_: s_,
                                    self.model.ac: action,
                                 } 
                                )
        return bonus

    def train(self, state, state_, actions, advantage, rewards, lstm_c, lstm_h, summary):

        fetches = [ self.model.summary_op, self.model.update_opt, self.model.global_step ]

        feed_dict = {
            self.model.st: state,
            self.model.s_: state_,
            self.model.ac: actions,
            self.model.adv: advantage,
            self.model.er: rewards,
            self.model.state_in[0]: lstm_c,
            self.model.state_in[1]: lstm_h,
        }

        fetched = self.model.sess( fetches, feed_dict )

        summary.add_summary( fetched[0], fetched[-1] )
        summary.flush()

    def sync(self):
        self.model.sess( self.model.sync )

class DQNLSTMCuriosityAgent():

    def __init__(self, model, state_size):
        self.model = model
        self.state_size = state_size

    def reset(self):
        return np.zeros( [ 1, self.state_size ] ), np.zeros( [ 1, self.state_size ] )

    def predict(self, observation, state, prior_act, summary):

        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if summary:
            actions, states, sm =\
            self.model.sess( 
                            [ self.model.q_value, self.model.state_out, self.model.summary ],
                            {
                                self.model.st: s,
                                self.model.a_: prior_act,
                                self.model.state_in[0]: state[0],
                                self.model.state_in[1]: state[1]
                            }
                            )

            return actions, states, sm
        else:
            actions, states =\
            self.model.sess( 
                            [ self.model.q_value, self.model.state_out ],
                            {
                                self.model.st: s,
                                self.model.a_: prior_act,
                                self.model.state_in[0]: state[0],
                                self.model.state_in[1]: state[1]
                            }
                            )

            return actions, states, None

    def get_action(self, actions):

        if np.random.uniform() < 0.9:
            # forward feed the observation and get q value for every actions
            return np.argmax( actions )
        else:
            return random.randint( 0, len(actions) - 1 )

    def pred_bonus(self, last_state, state, action):
        
        s = last_state[np.newaxis, :]
        s_ = state[np.newaxis, :]
        
        bonus = self.model.sess( self.model.bonus,
                                 {
                                    self.model.st: s,
                                    self.model.s_: s_,
                                    self.model.ac: action,
                                 } 
                                )
        return bonus

    def train(self, states, next_states, actions, p_actions, rewards, cell, hidden, dones, summary):

        fetches = [ self.model.summary_op, self.model.update_opt, self.model.global_step ]

        rewards = np.clip( rewards, -1, 1 )

        feed_dict = {
            self.model.st: states,
            self.model.s_: next_states,
            self.model.ac: actions,
            self.model.a_: p_actions,
            self.model.er: rewards,
            self.model.state_in[0]: cell,
            self.model.state_in[1]: hidden,
        }

        fetched = self.model.sess( fetches, feed_dict )

        summary.add_summary( fetched[0], fetched[-1] )
        summary.flush()

    def sync(self):
        self.model.sess( self.model.sync )

class ACExplorerThread(threading.Thread):
   
    def __init__(self, env, model, num_local_steps):
        
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)  # ideally, should be 1. Mostly doesn't matter in our case.
        
        self.num_local_steps = num_local_steps
        self.env = env
        self.model = model
        self.summary_writer = None

    def start_runner(self, summary_writer):
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        self._run()

    def _run(self):

        self.model.steps = 0

        if self.model.model.train:

            rollout_provider = ac_env_runner_explorer( self.env, self.model, self.summary_writer, self.num_local_steps )
            
            while True:
                self.queue.put( next( rollout_provider ), timeout = 600.0 )            
        
        else:
            while True:
                ac_env_runner_eval( self.env, self.model, self.summary_writer, self.num_local_steps )

class ACTrainerThread(threading.Thread):
   
    def __init__(self, model, gamma, _lambda, reawrd_clip):
        
        threading.Thread.__init__(self)       
        self.explorer = model
        self.agent = model.model
        self.gamma = gamma
        self._lambda = _lambda
        self.reawrd_clip = reawrd_clip
        self.summary_writer = None

    def start_runner(self, summary_writer):
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        self._run()

    def _run(self):

        while True:
            time.sleep(2)
            values = self._process()
            try:
                self.agent.train( *values, self.summary_writer )
            except Exception as e:
                print('\n\n== Error on {} ==\n\n{}\n\n================================='.format( self.agent.model.name, e ) )
            self.agent.sync()

    def _pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        rollout = self.explorer.queue.get( timeout = 600.0 )
        while not rollout.terminal:
            try:                    
                rollout.extend( self.explorer.queue.get_nowait() )
            except queue.Empty:
                break
        print(rollout.size())
        return rollout

    def _process(self):
        
        rollout = self._pull_batch_from_queue()
        
        batch = ac_process_rollout( rollout, gamma = self.gamma, lambda_ = self._lambda, clip = self.reawrd_clip )

        state = batch.si[:-1]
        state_ = batch.si[1:]
        actions = np.argmax( batch.a.squeeze(1), axis = 1 )
        advantage = batch.adv
        rewards = batch.r
        lstm_c = np.concatenate( [ x[0] for x in batch.features ], axis = 0 )
        lstm_h = np.concatenate( [ x[1] for x in batch.features ], axis = 0 )

        return state, state_, actions, advantage, rewards, lstm_c, lstm_h

class DQNExplorerThread(threading.Thread):
   
    def __init__(self, env, model, num_local_steps, gamma):
        
        threading.Thread.__init__(self)
        self.queue = queue.Queue(100)  # ideally, should be 1. Mostly doesn't matter in our case.
        
        self.num_local_steps = num_local_steps
        self.env = env
        self.model = model
        self.gamma = gamma
        self.summary_writer = None

    def start_runner(self, summary_writer):
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        self._run()

    def _run(self):

        self.model.steps = 0
        if self.model.model.train:
            dqn_env_runner_explorer( self.env, self.model, self.summary_writer, self.num_local_steps, self.gamma )        
        else:
            dqn_env_runner_eval( self.env, self.model, self.summary_writer, self.num_local_steps )

class DQNTrainerThread(threading.Thread):
   
    def __init__(self, model, gamma, reawrd_clip, update_steps):
        
        threading.Thread.__init__(self)       
        self.explorer = model
        self.agent = model.model
        self.gamma = gamma
        self.reawrd_clip = reawrd_clip
        self.summary_writer = None
        self.update_steps = update_steps

    def start_runner(self, summary_writer):
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        self._run()

    def _run(self):

        steps = 0
        while True:
            try:
                values = self._process()
                self.agent.train( *values, self.summary_writer )
            except Exception as e:
                print('\n\n== Error on {} ==\n\n{}\n\n================================='.format( self.agent.model.name, e ) )
            if steps % self.update_steps == 0:
                print("====================== Sync Target =====================")
                self.agent.sync()
            steps += 1

    def _process(self):

        states, next_states, actions, p_actions, rewards, bonus, t_states, dones, size = self.agent.model.memory.balanced_sample()
        
        t_states = np.transpose( t_states, axes = [ 1, 0, 3, 2 ] )
        t_states = t_states.reshape( t_states.shape[0:3] )
        
        rewards = np.sign( rewards.astype(np.float32) + bonus.astype(np.float32) )
        
        print( "Batch Size: {}".format( size ) )
        return states, next_states, actions, p_actions, rewards, t_states[0], t_states[1], dones

def ac_process_rollout(rollout, gamma, lambda_=1.0, clip=0.0):
    """
    Given a rollout, compute its returns and the advantage.
    """
    # collecting transitions
    batch_si = np.asarray(rollout.states + [rollout.end_state])
    batch_a = np.asarray(rollout.actions)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v1 = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping

    if clip != 0.0:
        rewards_plus_v1 = np.clip( rewards_plus_v1, -clip, clip )

    rewards_plus_v2 = np.asarray(rollout.bonuses + [0])
    rewards_plus_v = rewards_plus_v1 + rewards_plus_v2
        
    batch_r = discount( rewards_plus_v, gamma )[:-1]  # value network target

    # collecting target for policy network
    rewards1 = np.asarray(rollout.rewards)
    if clip != 0.0:
        rewards1 = np.clip( rewards1, -clip, clip )

    rewards2 = np.asarray(rollout.bonuses)    
    rewards = rewards1 + rewards2
        
    vpred_t = np.asarray( rollout.values + [ rollout.r ] )    
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * ( vpred_t[1:] - vpred_t[:-1] ).squeeze()
    batch_adv = discount( delta_t, gamma * lambda_ )

    features = rollout.features

    return Batch( batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features )

def ac_env_runner_eval(env, model, summary_writer, num_local_steps):
    
    last_state = env.reset()
    last_features = model.reset()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    stepAct = 0
    agent = model.model
    
    while True:

        terminal_end = False

        for _ in range(num_local_steps):

            # run policy
            fetched = model.predict( last_state, last_features, [ stepAct ], model.steps%100 == 0 )
            action, value_, features, sm = fetched[0], fetched[1], fetched[2], fetched[3]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = model.get_action( action )            
            state, reward, terminal, info = env.step(stepAct)
            state = state

            # render
            if agent.render:
                env.render()

            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features
                        
            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            
            if terminal or length >= timestep_limit:                
                
                print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
                
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                
                length = 0
                rewards = 0
                terminal_end = True

                last_features = model.reset() 
                
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()

            if info:
                summaries = []
                for k, v in info.items():
                    try:
                        summaries.append( tf.Summary.Value( tag = k, simple_value = float(v) ) )
                    except: pass
                if terminal:
                    summaries.append( tf.Summary.Value( tag = 'global/episode_value', simple_value = float( values ) ) )
                    values = 0
                                   
                summary = tf.Summary( value = summaries )
                summary_writer.add_summary( summary, model.steps )

                if not sm is None:
                    summary_writer.add_summary( sm, model.steps )
                summary_writer.flush()

            model.steps += 1

            if terminal_end:
                break

def ac_env_runner_explorer(env, model, summary_writer, num_local_steps):
    
    last_state = env.reset()
    last_features = model.reset()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    stepAct = 0

    agent = model.model
    ep_bonus = 0
    life_bonus = 0

    while True:

        terminal_end = False
        rollout = MemoryRollout()

        for xs in range(num_local_steps):

            # run policy
            fetched = model.predict( last_state, last_features, [ stepAct ], model.steps%100 == 0 )
            action, value_, features, sm = fetched[0], fetched[1], fetched[2], fetched[3]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = model.get_action( action )            
            state, reward, terminal, info = env.step( stepAct )
            state = state

            # render
            if agent.render:
                env.render()
            
            # reset reward
            if agent.noReward:
                reward = 0.
            

            if agent.curiosity:
                bonus = model.pred_bonus( last_state, state, [ stepAct ] )                
            else:
                bonus = [ 0.0 ] 
            
            curr_tuple = [ last_state, action, reward, value_, terminal, last_features, bonus[0], state ]
            
            life_bonus += bonus[0]
            ep_bonus += bonus[0]

            # collect the experience
            if model.steps % 10 == 0:
                rollout.add(*curr_tuple)
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            
            if terminal or length >= timestep_limit:                
                
                # prints summary of each life if envWrap==True else each game
                if agent.curiosity and not model.model.train:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (rewards, length, life_bonus))
                    life_bonus = 0
                else:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
                
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                
                length = 0
                rewards = 0
                terminal_end = True

                last_features = model.reset() # reset lstm memory
                
                # TODO: don't reset when gym timestep_limit increases, bootstrap -- doesn't matter for atari?
                # reset only if it hasn't already reseted
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()

            if info:
                # summarize full game including all lives (even if envWrap=True)
                summaries = []
                for k, v in info.items():
                    try:
                        summaries.append( tf.Summary.Value( tag = k, simple_value = float(v) ) )
                    except: pass
                if terminal:
                    summaries.append( tf.Summary.Value( tag = 'global/episode_value', simple_value = float( values ) ) )
                    values = 0
                    if agent.curiosity:
                        summaries.append( tf.Summary.Value( tag = 'global/episode_bonus', simple_value = float( ep_bonus ) ) )
                        ep_bonus = 0
                
                summary = tf.Summary( value = summaries )
                summary_writer.add_summary( summary, model.steps )

                if not sm is None:
                    summary_writer.add_summary( sm, model.steps )
                summary_writer.flush()

            model.steps += 1

            if terminal_end:
                break
        
        if not terminal_end:
            rollout.r = model.value( last_state, last_features, [ stepAct ] )

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

def dqn_env_runner_eval(env, model, summary_writer, num_local_steps):
    
    last_state = env.reset()
    last_features = model.reset()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    stepAct = 0
    agent = model.model
    
    while True:

        terminal_end = False

        for _ in range(num_local_steps):

            # run policy
            fetched = model.predict( last_state, last_features, [ stepAct ], model.steps%100 == 0 )
            action, value_, features, sm = fetched[0], fetched[1], fetched[2], fetched[3]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = model.get_action( action )            
            state, reward, terminal, info = env.step(stepAct)
            state = state

            # render
            if agent.render:
                env.render()

            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features
                        
            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            
            if terminal or length >= timestep_limit:                
                
                print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
                
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                
                length = 0
                rewards = 0
                terminal_end = True

                last_features = model.reset() 
                
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()

            if info:
                summaries = []
                for k, v in info.items():
                    try:
                        summaries.append( tf.Summary.Value( tag = k, simple_value = float(v) ) )
                    except: pass
                if terminal:
                    summaries.append( tf.Summary.Value( tag = 'global/episode_value', simple_value = float( values ) ) )
                    values = 0
                                   
                summary = tf.Summary( value = summaries )
                summary_writer.add_summary( summary, model.steps )

                if not sm is None:
                    summary_writer.add_summary( sm, model.steps )
                summary_writer.flush()

            model.steps += 1

            if terminal_end:
                break

def dqn_env_runner_explorer(env, model, summary_writer, num_local_steps, gamma):
    
    last_state = env.reset()
    p_state = list( [ last_state.copy() for _ in range( 3 ) ] )
    p_actions = list( [ 0 for _ in range( 3 ) ] )
    last_features = model.reset()  # reset lstm memory
    length = 0
    rewards = 0

    agent = model.model
    ep_bonus = 0
    life_bonus = 0

    while True:

        terminal_end = False

        for xs in range(num_local_steps):
            
            # get prior action
            p_a = np.eye(agent.action_size)[p_actions[-1]][np.newaxis,:]

            # run policy
            # fetched = model.predict( last_state, last_features, [ stepAct ], model.steps%100 == 0 )
            fetched = model.predict( last_state, last_features, p_a, False )
            action, features, sm = fetched[0], fetched[1], fetched[2]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = model.get_action( action )            
            state, reward, terminal, info = env.step( stepAct )

            # reset reward
            if agent.noReward:
                reward = 0.

            if agent.curiosity:
                bonus = model.pred_bonus( last_state, state, [ stepAct ] )                
            else:
                bonus = [ 0.0 ] 
            
            # delay last state
            p_s = p_state.pop(0)
            p_state.append( last_state.copy() )

            # accumulate actions 
            p_actions.pop(0)
            p_actions.append( stepAct )
            p_a = np.eye(agent.action_size)[p_actions]
            p_a = np.clip( np.sum( p_a, axis = 0 ), 0, 1 )

            # s, s_, a, a_, er, ir, c  
            curr_tuple = [ state.copy(), p_s.copy(), stepAct, p_a.copy(), reward, bonus[0], list(last_features).copy(), terminal_end ]

             # render
            if agent.render:
                img = cv.cvtColor( np.hstack( ( p_s, state ) ), cv.COLOR_BGR2RGB )
                cv.startWindowThread()
                cv.namedWindow("mario")
                cv.imshow("mario", img)
                cv.waitKey(1)

            life_bonus += bonus[0]
            ep_bonus += bonus[0]

            # collect the experience
            # if model.steps % 10 == 0:
            agent.memory.add(*curr_tuple)
            
            rewards += reward
            length += 1

            last_state = state
            last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            
            if terminal or length >= timestep_limit:
                
                # prints summary of each life if envWrap==True else each game
                if agent.curiosity and not model.model.train:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (rewards, info['x_pos'], life_bonus))
                    life_bonus = 0
                else:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, info['x_pos']))
                
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                
                length = 0
                rewards = 0
                terminal_end = True

                last_features = model.reset() # reset lstm memory
                
                # TODO: don't reset when gym timestep_limit increases, bootstrap -- doesn't matter for atari?
                # reset only if it hasn't already reseted
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()

            if info:
                # summarize full game including all lives (even if envWrap=True)
                summaries = []
                for k, v in info.items():
                    try:
                        summaries.append( tf.Summary.Value( tag = k, simple_value = float(v) ) )
                    except: pass
                if terminal:
                    if agent.curiosity:
                        summaries.append( tf.Summary.Value( tag = 'global/episode_bonus', simple_value = float( ep_bonus ) ) )
                        ep_bonus = 0
                
                summary = tf.Summary( value = summaries )
                summary_writer.add_summary( summary, model.steps )

                if not sm is None:
                    summary_writer.add_summary( sm, model.steps )
                summary_writer.flush()

            model.steps += 1

            if terminal_end:
                break

def discount(x, gamma):
    """       
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
