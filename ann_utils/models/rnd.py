import sys
sys.path.append('../../')

import tensorflow as tf

from ann_utils.fully_layer import FullyLayer as fc
from ann_utils.helper import gelu, prelu, to_float, softmax, cosine_loss

class RND:

    def __init__(self, s_encode_size, h_size, ddqn_model, name):

        self.name = name

        self.r_n = FullyLayer( s_encode_size, '{}_random_net'.format( name ), act = gelu )
        
        self.p_h = FullyLayer( h_size, '{}_p_h'.format( name ), act = gelu )
        self.p_o = FullyLayer( s_encode_size, '{}_p_o'.format( name ), act = None )
        
        self.ddqn_model = ddqn_model
   
    def build_training_graph(self, n_s, gamma, lr, soft_update=0.9):
        st, ac, er, s_ = self.ddqn_model._build_inputs()
        pred_train, ddqn_model_train, q, ddqn_model_loss, soft_update_op = \
            self._build_nets(  st, ac, er, s_, gamma, lr, soft_update, True )         
        return st, ac, er, s_, pred_train, ddqn_model_train, q, soft_update_op, ddqn_model_loss
    
    def _build_nets(self, st, ac, er, s_, gamma, lr, soft_update, is_training):
        
        # normal ddqn_model model
        _, xes_, q, q_, soft_update_op = self.ddqn_model._build_net( st, s_, soft_update, is_training )

        # fixed random net
        with tf.compat.v1.variable_scope("random_net"):
            rand_encode_s_ = self.r_n( xes_, is_training = False )

        # predictor
        ri, pred_train = self.__build_predictor( xes_, rand_encode_s_, lr, is_training )

        # ddqn_model loss
        ddqn_model_loss, ddqn_model_train = self._build_ddqn_model_optmizer( er, ri, gamma, q_, ac, q, lr )

        return pred_train, ddqn_model_train, q, ddqn_model_loss, soft_update_op

    def __build_predictor(self, s_, rand_encode_s_, lr, is_training):

        with tf.compat.v1.variable_scope("predictor"):
            x = self.p_h( s_, is_training = is_training )
            x = self.p_o( x,  is_training = is_training ) 

        with tf.name_scope("int_r"):
            ri = tf.reduce_sum( tf.square( rand_encode_s_ - x ), axis = 1 )  # intrinsic reward
        
        vars = [ x for x in tf.compat.v1.trainable_variables() if 'predictor' in x.name or 'backbone' in x.name ]        
        
        grads = tf.gradients( tf.reduce_mean( ri ), vars )
        grads_and_vars = list( zip( grads, vars ) )
        capped_gvs = [ ( tf.clip_by_value( grad, -6., 6. ), var ) for grad, var in grads_and_vars if grad != None ]        
        
        train_op = tf.train.MomentumOptimizer( lr, 0.9, name = "predictor_opt" ).apply_gradients( capped_gvs )
        
        return ri, train_op

    def _build_ddqn_model_optmizer(self, re, ri, gamma, q_, ac, q, lr):
        
        with tf.compat.v1.variable_scope('q_target'):
            q_target = re + ri + gamma * tf.reduce_max( q_, axis = 1, name = "Qmax_s_" )# ddqn_model error with intrinsic reward
            # q_target = re + ri + gamma * q_# ddqn_model error with intrinsic reward

        with tf.compat.v1.variable_scope('q_wrt_a'):
            a_indices = tf.stack( [ tf.range( tf.shape( ac )[0], dtype = tf.int32), ac ], axis = 1 )
            q_wrt_a = tf.gather_nd( params = q, indices = a_indices )

        loss = tf.losses.huber_loss( labels = q_target, predictions = q_wrt_a ) # ddqn_model error
        
        vars = self.ddqn_model.get_eval_variables()
        
        grads = tf.gradients( loss, vars )
        grads_and_vars = list( zip( grads, vars ) )
        capped_gvs = [ ( tf.clip_by_value( grad, -6., 6. ), var ) for grad, var in grads_and_vars if grad != None ]        
        
        train_op = tf.train.MomentumOptimizer( lr, 0.9, name = "ddqn_model_opt" ).apply_gradients( capped_gvs )
        
        return loss, train_op

class Intrinsic_Curiosity_Module(object):

    def __init__(self, 
                 rnds_par, 
                 rndsa_par,
                 inv_par,
                 icm_par,
                 dp=0.25,
                 hidden_act=prelu,
                 out_act=None
                 ):

        # fixed rnds
        self.rnds_f_h, self.rnds_f_o = self.__create_variables( rnds_par, 'f' )
        
        # dynamic rnds
        self.rnds_h, self.rnds_o = self.__create_variables( rnds_par )

        # fixed rndsa
        self.rndsa_f_h, self.rndsa_f_o = self.__create_variables( rndsa_par, 'f' )
        # dynamic rndsa
        self.rndsa_h, self.rndsa_o = self.__create_variables( rndsa_par )

        # inverse
        self.inverse_h, self.inverse_o = self.__create_variables( inv_par )

        # icm
        self.icm_h, self.icm_o = self.__create_variables( icm_par )
                        
    def __call__(self, s, s_, ac, ac_space, is_training=False, summary=False):
        
        # rnds
        ir_, pred_vars_ = self.__build_random_network_distilation_new_state( s_, is_training, summary )

        # rndsa
        ir, pred_vars = self.__build_random_network_distilation_state_vs_action( s, ac, ac_space, is_training, summary )

        # inverse
        inverse_pred, inverse_vars = self.__build_inverse_network( ac, s, s_, is_training, summary )

        # icm
        icm_ir, icm_vars = self.__build_intrinsic_network( ac, s, s_, is_training, summary )

        return ( ir_, ir, icm_ir, inverse_pred,
                 pred_vars_, pred_vars, icm_vars, inverse_vars )

    def __create_variables(self, par, pre=''):

        h = [ 
            fc( x, 
                '{}{}_h'.format( pre, i ), 
                act = par['h_act'], 
                dropout = par['dp'], 
                std = par['std'] ) 
            for i, x in enumerate( par['hidden'] )
        ]

        o = fc( par['size'], 
                '{}_o'.format( pre ), 
                dropout = par['dp'], 
                act = par['o_act'], 
                std = par['std'] )
        
        return h, o

    def __build_random_network_distilation_new_state(self, s_, is_training, summary=False):

        # fixed state random net
        with tf.compat.v1.variable_scope("rnds_random_net"):
            x = s_
            for n in self.rnds_f_h:
                x = n( x, is_training = False )
            rand_encode_s = self.rnds_f_o( x, is_training = False )

        # trainable state random net
        with tf.compat.v1.variable_scope("rnds_predictor"):
            x = s_
            for n in self.rnds_h:
                x = n( x, is_training = is_training )
            xs = self.rnds_o( x, is_training = is_training ) 
            
            ri = cosine_loss( rand_encode_s, xs )

        vars = [ x for x in tf.compat.v1.trainable_variables() if 'rnds_predictor' in x.name ]
        
        if summary:
            
            tf.summary.scalar( 'rnd_ir', tf.reduce_mean( ri ), family = 'curiosity' )

            for w in vars:
                tf.summary.histogram( family = 'rnds_var_', name = w.name.replace(':', '_'), values = w )
        
        return ri, vars

    def __build_random_network_distilation_state_vs_action(self, s, ac, ac_space, is_training, summary=False):
        
        # create satate and action
        one_hot_action = tf.one_hot( ac, ac_space )
        sa = tf.concat( [ s, one_hot_action ], axis = 1 )
                
        # fixed state_action random net
        with tf.compat.v1.variable_scope("rndsa_random_net"):
            x = sa
            for n in self.rndsa_f_h:
                x = n( x, is_training = False )
            rand_encode_s = self.rndsa_f_o( x, is_training = False )

        with tf.compat.v1.variable_scope("rndsa_predictor"):
            x = s
            for n in self.rndsa_h:
                x = n( x, is_training = is_training )
            xs = self.rndsa_o( x, is_training = is_training )
            
            ri = cosine_loss( rand_encode_s, xs )

        vars = [ x for x in tf.compat.v1.trainable_variables() if 'rndsa_predictor' in x.name ]
               
        if summary:
            
            tf.summary.scalar( 'rndsa_ir', tf.reduce_mean( ri ), family = 'curiosity' )

            for w in vars:
                tf.summary.histogram( family = 'rndsa_var_', name = w.name.replace(':', '_'), values = w )
        
        return ri, vars

    def __build_inverse_network(self, ac, s, s_, is_training, summary=False):

        x = tf.concat( [ s, s_ ], axis = 1 )

        with tf.compat.v1.variable_scope('inverse_netwrok'):

            for n in self.inverse_h:
                x = n( x, is_training = is_training )
            xs = self.inverse_o( x, is_training = is_training ) 
            
        vars = [ x for x in tf.compat.v1.trainable_variables() if 'inverse_netwrok' in x.name ]

        if summary:

            for w in vars:
                tf.summary.histogram( family = 'inverse_netwrok_var', name = w.name.replace(':', '_'), values = w )
        
        return xs, vars

    def __build_intrinsic_network(self, ac, s, s_, is_training, summary=False):

        one_hot_action = tf.one_hot( ac, self.inverse_o.size )
        x = tf.concat( [ one_hot_action, s ], axis = 1 )
        with tf.compat.v1.variable_scope("icm"):
            
            for n in self.icm_h:
                x = n( x, is_training = is_training )
            xs = self.icm_o( x, is_training = is_training, size = s_.shape[1] ) 
            
            ri = cosine_loss( s_, xs )

        vars = [ x for x in tf.compat.v1.trainable_variables() if 'icm' in x.name ]
               
        if summary:
            tf.summary.scalar( 'icm_ir', tf.reduce_mean( ri ), family = 'curiosity' )
            for w in vars:
                tf.summary.histogram( family = 'icm_var', name = w.name.replace(':', '_'), values = w )
        
        return ri, vars
