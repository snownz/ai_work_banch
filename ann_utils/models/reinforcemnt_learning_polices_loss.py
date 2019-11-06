import tensorflow as tf
from ann_utils.helper import to_float

class Qlearning(object):

    def __init__(self, gamma=0.1, entropy_beta=0.01):
        self.gamma = gamma
        self.entropy_beta = entropy_beta

    def __call__(self, i_er, q_, q, ac, summary):

        q_target = ( i_er + ( self.gamma * tf.reduce_max( q_, axis = 1 ) ) )
                
        a_indices = tf.stack( [ tf.range( tf.shape( ac )[0], dtype = tf.int32 ), ac ], axis = 1 )
        q_wrt_a = tf.gather_nd( params = q, indices = a_indices ) 

        # compute loss
        # loss = tf.sqrt( tf.square( q_target - q_wrt_a ) )
        loss = tf.losses.huber_loss( labels = q_target, predictions = q_wrt_a )
        
        return loss

class AC(object):

    def __init__(self, entropy_beta=0.01):
        self.entropy_beta = entropy_beta

    def __call__(self, logits, vf, ac, r, adv, summary):

        # Computing a3c loss: https://arxiv.org/abs/1506.02438        
        log_prob_tf = tf.nn.log_softmax( logits )
        prob_tf = tf.nn.softmax( logits )
        one_hot_action = tf.one_hot( ac, logits.shape[-1] )

        # 1) the "policy gradients" loss:  its derivative is precisely the policy gradient
        # notice that self.ac is a placeholder that is provided externally.
        # adv will contain the advantages, as calculated in process_rollout
        with tf.compat.v1.variable_scope( 'pi_loss' ):
            p1 = tf.multiply( log_prob_tf, one_hot_action, name = 'log_prob_tf_x_one_hot_action' )            
            # pi_loss = - tf.reduce_mean( tf.reduce_sum( p1, 1 ) ) * tf.reduce_mean( adv * to_float( adv >= 0.0 ) * to_float( adv <= 100.0 ) ) # Eq (19)
            pi_loss = -tf.reduce_mean( p1 * tf.stop_gradient( adv )[:,tf.newaxis] ) # Eq (19)
            # pi_loss = - tf.reduce_mean( tf.reduce_sum( p1, 1 ) * adv )  # Eq (19)

        tf.summary.histogram( family = 'ac_values', name = 'log_prob_tf', values = log_prob_tf )
        tf.summary.histogram( family = 'ac_values', name = 'prob_tf', values = prob_tf )
        tf.summary.histogram( family = 'ac_values', name = 'advantage', values = adv )
        tf.summary.histogram( family = 'ac_values', name = 'r', values = r )
        tf.summary.histogram( family = 'ac_values', name = 'vf', values = vf )
        
        # 2) loss of value function: l2_loss = (x-y)^2/2
        with tf.compat.v1.variable_scope( 'vf_loss' ):
            vf_loss = 0.5 * tf.reduce_mean( tf.square( r - vf ) )  # Eq (28)
        
        # 3) entropy to ensure randomness
        with tf.compat.v1.variable_scope( 'entropy' ):
            # entropy = self.entropy_beta *  tf.reduce_mean( ( - tf.reduce_sum( prob_tf * log_prob_tf, axis = 1 ) ) )
            entropy = -tf.reduce_mean( prob_tf * log_prob_tf )
        
        # final a3c loss: lr of critic is half of actor
        with tf.compat.v1.variable_scope( 'loss' ):
            loss = 0.5 * vf_loss + pi_loss - entropy * self.entropy_beta

        if summary:
            tf.summary.scalar( 'pi_loss', pi_loss, family = 'ac' )
            tf.summary.scalar( 'vf_loss', vf_loss, family = 'ac' )
            tf.summary.scalar( 'entropy', entropy, family = 'ac' )
            tf.summary.scalar( 'loss', loss, family = 'ac' )

        return loss