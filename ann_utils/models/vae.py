import sys
sys.path.append('../../')

import tensorflow as tf
import numpy as np

from ann_utils.helper import maxpool2d, avgpool2d, flatten, l2, noise, mse_loss, upsampling2d

from ann_utils.conv_layer import Conv2DLayer as conv 
from ann_utils.conv_layer import Deconv2DLayer as dconv
from ann_utils.conv_layer import SeparableConv2DLayer as sconv
from ann_utils.models.squeeze_n_excitation import SqueezeNExcitation as senet
from ann_utils.fully_layer import FullyLayer as fc
from ann_utils.self_attention import Self_Attention_Multi_Head_3D_GB as attn_mh

def _reparameterize(mu, logvar, SAMPLES):
        
    samples_z = []
    std = 0.5 * tf.exp( logvar )
    for _ in range(SAMPLES):
        eps = tf.random_normal( shape = tf.shape( std ), mean = 0, stddev = 1, dtype = tf.float32 )
        z = mu + ( eps * std )
        samples_z.append( z )  
    return samples_z

class Self_Attention_Input_Simple_Image_Vae( object ):

    def __init__( self, 
                  name, 
                  size=64, 
                  ch=3, 
                  dp=0.0, 
                  lr=1e-3,
                  decay_steps=1000,
                  decay_keep=.90,
                  bn=False, 
                  num_heads=8,
                  act=tf.nn.relu
                ):

        self.initial_learning_rate = lr
        self.decay_steps = decay_steps
        self.decay_keep = decay_keep
        self.size = size
        
        self.attn = attn_mh( ch, name, ch, num_heads, out_act = tf.nn.sigmoid )

        # encoder
        self.ec1 = conv( 16, 3, 2, "{}_c1".format( name ), dropout = dp, bn = bn, act = act, bias = True )
        self.ec2 = conv( 16, 3, 2, "{}_c2".format( name ), dropout = dp, bn = bn, act = act, bias = True )
        self.ec3 = conv( 16, 3, 2, "{}_c3".format( name ), dropout = dp, bn = bn, act = act, bias = True )
        self.ec4 = conv( 16, 3, 2, "{}_c4".format( name ), dropout = dp, bn = bn, act = act, bias = True )

        self.ef1 = fc( 1024, "{}_f1".format( name ), dropout = dp, act = act, bias = True )
        self.ef11 = fc( size, "{}_f11".format( name ), dropout = dp, act = None, bias = True )
        
        self.ef2 = fc( 1024, "{}_f2".format( name ), dropout = dp, act = act, bias = True )
        self.ef21 = fc( size, "{}_f21".format( name ), dropout = dp, act = None, bias = True )

        # decoder
        self.df2 = fc( 1024, "{}_f1".format( name ), dropout = dp, act = act, bias = True )
        self.df21 = fc( None, "{}_f11".format( name ), dropout = dp, act = act, bias = True )
       
        self.dc1 = dconv( 16, 3, 2, "{}_c1".format( name ), dropout = dp, bn = bn, act = tf.nn.relu, bias = True )
        self.dc2 = dconv( 16, 3, 2, "{}_c2".format( name ), dropout = dp, bn = bn, act = tf.nn.relu, bias = True )
        self.dc3 = dconv( 16, 3, 2, "{}_c3".format( name ), dropout = dp, bn = bn, act = tf.nn.relu, bias = True )
        self.dc4 = dconv( ch, 3, 2, "{}_c4".format( name ), dropout = dp, bn = bn, act = tf.nn.sigmoid, bias = True )
    
    def build_variables(self, summary=False):

        with tf.compat.v1.variable_scope('Self_Attention_Input_Simple_Image_Vae'):
            self.global_step = tf.compat.v1.get_variable( 'vae_global_step', [], 
                                                        initializer = tf.constant_initializer(0), 
                                                        trainable = False )
            
            self.learning_rate = tf.compat.v1.train.exponential_decay( self.initial_learning_rate, 
                                                                    self.global_step, 
                                                                    self.decay_steps, 
                                                                    self.decay_keep, 
                                                                    staircase = False )
      
        if summary:
            tf.summary.scalar( 'global_step', self.global_step, family = 'vae' )
            tf.summary.scalar( 'learning_rate', self.learning_rate, family = 'vae' )

    def _build_encoder(self, x, is_training=False, summary=False):
        
        attn_encoded, attn_vars = self.attn( x, True, summary, 2 )
        attn_encoded = ln( x + attn_encoded )

        encoded1 = attn_encoded
        encoded2 = self.ec1( encoded1, is_training = is_training )
        encoded3 = self.ec2( encoded2, is_training = is_training )
        encoded4 = self.ec3( encoded3, is_training = is_training )
        encoded5 = self.ec4( encoded4, is_training = is_training )

        features = flatten( encoded5 )

        z_mu = self.ef1( features, is_training )
        z_mu = self.ef11( z_mu, is_training, self.size )

        z_log_sigma_sq = self.ef2( features, is_training )
        z_log_sigma_sq = self.ef21( z_log_sigma_sq, is_training, self.size )

        if summary:

            tf.summary.image( family = 'vae_values', name = 'encoder_input', tensor = attn_encoded, max_outputs = 1 )

            tf.summary.histogram( family = 'vae_layers', name = "e_l1", values = encoded1 )
            tf.summary.histogram( family = 'vae_layers', name = "e_l2", values = encoded2 )
            tf.summary.histogram( family = 'vae_layers', name = "e_l4", values = encoded3 )
            tf.summary.histogram( family = 'vae_layers', name = "e_l4", values = encoded4 )
            tf.summary.histogram( family = 'vae_layers', name = "e_l5", values = encoded5 )
            tf.summary.histogram( family = 'vae_layers', name = "mu", values = z_mu )
            tf.summary.histogram( family = 'vae_layers', name = "log_sigma", values = z_log_sigma_sq )

        return attn_encoded, attn_vars, z_mu, z_log_sigma_sq, tf.shape(encoded5), features.shape[1] 

    def _build_decoder(self, z, size, shape, is_training=False, summary=False):
                
        decoded1 = self.df2( z, is_training )
        decoded2 = self.df21( decoded1, is_training, size )

        # reshape
        decoded = tf.reshape( decoded2, shape )        
        
        decoded3 = self.dc1( decoded, is_training = is_training )
        decoded4 = self.dc2( decoded3, is_training = is_training )
        decoded5 = self.dc3( decoded4, is_training = is_training )
        decoded6 = self.dc4( decoded5, is_training = is_training )

        if summary:

            tf.summary.image( family = 'vae_values', name = 'decoder_output', tensor = decoded6, max_outputs = 1 )

            tf.summary.histogram( family = 'vae_layers', name = "d_l1", values = decoded1 )
            tf.summary.histogram( family = 'vae_layers', name = "d_l2", values = decoded2 )
            tf.summary.histogram( family = 'vae_layers', name = "d_l3", values = decoded3 )
            tf.summary.histogram( family = 'vae_layers', name = "d_l4", values = decoded4 )
            tf.summary.histogram( family = 'vae_layers', name = "d_l5", values = decoded5 )
            tf.summary.histogram( family = 'vae_layers', name = "d_l6", values = decoded6 )

        return decoded6

    def __call__(self, x, is_training=False, summary=False, samples=1):
        
        # encoder
        with tf.compat.v1.variable_scope('vae_encode'):
            attn, attn_vars, mu, logvar, shape, size = self._build_encoder( x, True, summary )
            # norm_attn = ( attn - tf.reduce_min( attn ) ) / ( tf.reduce_max( attn ) - tf.reduce_min( attn ) )

        # lattent spoace
        zs = _reparameterize( mu, logvar, samples )

        if not is_training:
            return zs[0]

        # decoder
        with tf.compat.v1.variable_scope('vae_decode'):
            decoded = [ self._build_decoder( z, size, shape, True, summary ) for z in zs ]

        with tf.compat.v1.variable_scope('vae_loss'):
            BCE = 0
            for recon_x in decoded:
                # BCE += tf.losses.softmax_cross_entropy( logits = recon_x, onehot_labels = tf.reduce_mean( x, axis = 3, keepdims = True ) )
                BCE += mse_loss( x, recon_x )
                # exp = ( flatten( x ) - flatten( recon_x ) ) ** 2
                # BCE += tf.reduce_sum( exp, axis = 1 )
            BCE /= samples

            KLD = -0.5 * tf.reduce_sum( 1.0 + logvar - tf.pow( mu, 2 ) - tf.exp( logvar ), axis = 1 )
            KLD /= self.size
                       
        encoder_vars = [ w for w in tf.compat.v1.trainable_variables() if 'vae_encode' in w.name and not '_attn_' in w.name ]
        decoder_vars = [ w for w in tf.compat.v1.trainable_variables() if 'vae_decode'  in w.name ]

        encoder_update_ops = [ w for w in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'vae_encode' in w.name and not '_attn_' in w.name ]
        attn_update_ops = [ w for w in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'vae_encode' in w.name and '_attn_' in w.name ]
        decoder_update_ops = [ w for w in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'vae_decode'  in w.name ]

                
        if summary:

            tf.summary.scalar( family = 'vae', name = 'recon_loss', tensor = tf.reduce_mean( BCE ) )
            tf.summary.scalar( family = 'vae', name = 'latent_loss', tensor = tf.reduce_mean( KLD ) )

            for w in encoder_vars:
                tf.summary.histogram( family = 'vae_encode_var', name = w.name, values = w )
            
            for w in decoder_vars:
                tf.summary.histogram( family = 'vae_decode_var', name = w.name, values = w )

        return zs[0], decoded[0], BCE, KLD, encoder_vars, attn_vars, decoder_vars, encoder_update_ops, attn_update_ops, decoder_update_ops

def create_layers( index, par ):
    
    if par['type'] == 'conv':
        return conv( par['out'], 
                     par['k'], 
                     par['s'], 
                     "conv_layer_{}".format(index), 
                     dropout = par['dp'], 
                     bn = par['bn'], 
                     act = par['act'], 
                     bias = par['bias'] )

    if par['type'] == 'se':
        return senet( "se_layer_{}".format(index), act = par['act'] )

    if par['type'] == 'sconv':
        return sconv( par['out'], 
                      par['k'], 
                      par['s'], 
                      "sconv_layer_{}".format(index), 
                      dropout = par['dp'], 
                      bn = par['bn'], 
                      act = par['act'] )

    if par['type'] == 'dconv':
        return dconv( par['out'], 
                     par['k'], 
                     par['s'], 
                     "dconv_layer_{}".format(index), 
                     dropout = par['dp'], 
                     bn = par['bn'], 
                     act = par['act'], 
                     bias = par['bias'] )
    
    if par['type'] == 'fully':
        return fc( par['size'], 
                   "fully_layer_{}".format( index ), 
                   dropout = par['dp'], 
                   act = par['act'], 
                   bias = par['bias'] )

class Simple_Image_Vae( object ):

    def __init__( self, 
                  conv_layers,
                  dconv_layers,
                  size,
                  dp=0.0,
                  act=tf.nn.relu ):

        self.size = size

        # encoder
        self.encoder = [ create_layers( i, par ) for i, par in enumerate( conv_layers ) ]
        
        self.ef1 = fc( None, "mu_h", dropout = dp, act = act, bias = True )
        self.ef11 = fc( size, "mu", dropout = 0.0, act = None, bias = True )
        
        self.ef2 = fc( None, "logvar_h", dropout = dp, act = act, bias = True )
        self.ef21 = fc( size, "logvar", dropout = dp, act = None, bias = True )

        # decoder
        self.df1 = fc( size, "resample_h", dropout = dp, act = act, bias = True )
        self.df11 = fc( None, "resample", dropout = dp, act = None, bias = True )

        self.decoder = [ create_layers( i, par ) for i, par in enumerate( dconv_layers ) ]    

    def _build_encoder(self, x, is_training=False, summary=False):
        
        encoded = x
        for n in self.encoder:
            encoded = n( encoded, is_training = is_training )

        features = flatten( encoded )

        z_mu = self.ef1( features, is_training, 1024 )
        z_mu = self.ef11( z_mu, is_training )

        z_log_sigma_sq = self.ef2( features, is_training, 1024 )
        z_log_sigma_sq = self.ef21( z_log_sigma_sq, is_training )
          
        return z_mu, z_log_sigma_sq, tf.shape( encoded ), features.shape[1] 

    def _build_decoder(self, z, size, shape, is_training=False, summary=False):
        
        decoded1 = self.df1( z, is_training )
        decoded2 = self.df11( decoded1, is_training, size )

        # reshape
        decoded = tf.reshape( decoded2, shape )
        for n in self.decoder:
            decoded = n( decoded, is_training = is_training )
            decoded = upsampling2d( decoded, 2 )
       
        return decoded

    def __call__(self, x, is_training=False, summary=False, samples=1):
        
        # encoder
        with tf.compat.v1.variable_scope('vae_encode'):
            mu, logvar, shape, size = self._build_encoder( x, True, summary )
        encoder_vars = [ w for w in tf.trainable_variables() if 'vae_encode' in w.name ]

        # lattent space
        with tf.compat.v1.variable_scope('vae_reparameterize'):
            zs = _reparameterize( mu, logvar, samples )

        if not is_training:
            return mu, zs, encoder_vars

        # decoder
        with tf.compat.v1.variable_scope('vae_decode'):
            decoded = [ self._build_decoder( z, size, shape, True, i == 0 and summary ) for i, z in enumerate( zs ) ]
            dc = tf.concat( [ tf.expand_dims( x, axis = 0 ) for x in decoded ], axis = 0 )                       
        decoder_vars = [ w for w in tf.compat.v1.trainable_variables() if 'vae_decode' in w.name ]

        encoder_update_ops = [ w for w in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'vae_encode' in w.name  ]
        decoder_update_ops = [ w for w in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'vae_decode'  in w.name ]
                
        return mu, logvar, zs,\
               decoded,\
               encoder_vars, decoder_vars,\
               encoder_update_ops, decoder_update_ops,\
               shape, size

class Simple_Array_Vae( object ):

    def __init__( self, 
                  encode_layers,
                  decode_layers,
                  size,
                  ch=3,
                  dp=0.0,
                  act=tf.nn.relu ):

        self.size = size

        # encoder
        self.encoder = [ create_layers( i, par ) for i, par in enumerate( encode_layers ) ]
        
        self.ef1 = fc( None, "f1", dropout = dp, act = act, bias = False )
        self.ef11 = fc( size, "f11", dropout = 0.0, act = None, bias = False )
        
        self.ef2 = fc( None, "f2", dropout = dp, act = act, bias = False )
        self.ef21 = fc( size, "f21", dropout = dp, act = None, bias = False )

        self.decoder = [ create_layers( i, par ) for i, par in enumerate( decode_layers ) ]
        self.df1 = fc( size, "f1", dropout = dp, act = act, bias = False ) 

    def _build_encoder(self, x, is_training=False, summary=False):
        
        encoded = x
        for n in self.encoder:
            encoded = n( encoded, is_training = is_training )

        features = encoded

        z_mu = self.ef1( features, is_training, ( features.shape[-1] + self.size ) // 2 )
        z_mu = self.ef11( z_mu, is_training )

        z_log_sigma_sq = self.ef2( features, is_training, ( features.shape[-1] + self.size ) // 2 )
        z_log_sigma_sq = self.ef21( z_log_sigma_sq, is_training )

        return z_mu, z_log_sigma_sq, x.shape[1] 

    def _build_decoder(self, z, size, is_training=False, summary=False):
        
        decoded = z
        for n in self.decoder:
            decoded = n( decoded, is_training = is_training )
        decoded = self.df1( decoded, is_training = is_training, size = size )
       
        return decoded

    def __call__(self, x, is_training=False, summary=False, samples=1):
        
        # encoder
        with tf.compat.v1.variable_scope('vae_encode'):
            mu, logvar, size = self._build_encoder( x, is_training, summary )

        # lattent spoace
        zs = _reparameterize( mu, logvar, samples )

        zc = tf.concat( [ tf.expand_dims( x, axis = 0 ) for x in zs ], axis = 0 )

        if not is_training:
            return zc

        # decoder
        with tf.compat.v1.variable_scope('vae_decode'):
            decoded = [ self._build_decoder( z, size, is_training, i == 0 and summary ) for i, z in enumerate( zs ) ]
            dc = tf.concat( [ tf.expand_dims( x, axis = 0 ) for x in decoded ], axis = 0 )
                       
        encoder_vars = [ w for w in tf.compat.v1.trainable_variables() if 'vae_encode' in w.name  ]
        decoder_vars = [ w for w in tf.compat.v1.trainable_variables() if 'vae_decode'  in w.name ]
                
        return mu, logvar,\
               zc, dc,\
               decoded,\
               encoder_vars, decoder_vars
