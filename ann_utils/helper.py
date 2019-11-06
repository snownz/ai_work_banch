import tensorflow as tf
import tensorflow.compat.v1.keras.layers as KL
import numpy as np

# =============================================== cast =============================================== 

def to_float(x):
    return tf.cast( x, tf.float32 )

# =============================================== costs =============================================== 

def binary_cross_entropy_loss(target, predict, eps=1e-12):
    return (-( target * tf.log( predict + eps ) + ( 1. - target ) * tf.log( 1. - predict + eps ) ))

def mse_loss( target, predict ):
    return tf.sqrt( tf.square( target - predict ) )

def huber_loss(x, delta=1.0):    
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def cosine_loss(A, B, keepdims=False):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum( tf.multiply( tf.nn.l2_normalize( A, 1 ), tf.nn.l2_normalize( B, 1 ) ), 1, keepdims = keepdims )
    loss = 1 - dotprod
    return loss

def l2(scale, vars):
    with tf.variable_scope('l2'):
        reg = tf.contrib.layers.l2_regularizer( scale = scale )
        return tf.contrib.layers.apply_regularization( reg, vars )
 
def l1(scale):
    return tf.contrib.layers.l1_regularizer( scale = scale )

def l21_norm(x):
    # Computes the L21 norm of a symbolic matrix W
    return tf.reduce_sum( tf.norm( x, axis = 1 ) )

def group_regularization(v, keepdims=False):
    # Computes a group regularization loss from a list of weight matrices corresponding
    const_coeff = lambda W: tf.sqrt( tf.cast( W.get_shape().as_list()[1], tf.float32 ) )
    return tf.reduce_sum( [ tf.multiply( const_coeff( W ), l21_norm( W ) ) for W in v ], keepdims = keepdims )

# =============================================== tools operations =============================================== 

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn( *shape ).astype( np.float32 )
        out *= tf.div_no_nan( std, np.sqrt( np.square( out ).sum( axis = 0, keepdims = True ) ) )
        return tf.Variable( out )
    return _initializer

def count_neurons(v):
    count_neurons = lambda W: tf.reduce_sum( tf.cast( tf.greater( tf.reduce_sum( tf.abs( W ), reduction_indices = [1] ), 10**-3 ), tf.float32 ) )
    return count_neurons( v )

# =============================================== optmizer operations =============================================== 

"""
From OpenAI
"""  
def flatgrad(loss, var_list, clip_norm=None):
    
    grads = tf.gradients( loss, var_list )
    
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

# =============================================== operations =============================================== 

def log2(x):
    """Log2"""
    return tf.log(x) / tf.log(2.0)

def flatten(x):
    shapes = x.get_shape().as_list()
    shapes = np.multiply.reduce( shapes[1:len(shapes)] )
    x = tf.reshape( x, [ -1, shapes ] )
    # print(x)
    return x

def hw_flatten(x):
    return tf.reshape( x, [ tf.shape(x)[0], x.shape[1]*x.shape[2], x.shape[3] ] )

def hw_flatten_multi_head(x):
    return tf.reshape( x, [ tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3], x.shape[4] ] )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def dropout(x, dp=0.5, name="dp"):

    with tf.variable_scope('dropout'):
        x = tf.nn.dropout( x, rate = dp )
        # print(x)
        return x

def concat(x, axis, name):
    x = tf.concat( x, axis = axis, name = name )
    # print(x)
    return x

def maxpool2d(x, k, s, padding='SAME'):

    with tf.variable_scope('maxpool2d'):
        if type(k) is tuple:
            ksize = [1, k[0], k[1], 1]
        else:
            ksize = [1, k, k, 1]

        if type(s) is tuple:
            strides = [1, s[0], s[1], 1]
        else:
            strides = [1, s, s, 1]

        x = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)
        # print(x)
        return x

def avgpool2d(x, k, s, padding='SAME'):
    
    with tf.variable_scope('avgpool2d'):
        if type(k) is tuple:
            ksize = [1, k[0], k[1], 1]
        else:
            ksize = [1, k, k, 1]

        if type(s) is tuple:
            strides = [1, s[0], s[1], 1]
        else:
            strides = [1, s, s, 1]

        x = tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding=padding)
        # print(x)
        return x

def upsampling2d(x, size):
    return KL.UpSampling2D( size = size )( x )

def upsampling1d(x, size):
    return KL.UpSampling1D( size = size )( x )

def global_average_pool_spartial(x):
    with tf.variable_scope('global_average_pool_spartial'):
        c = x.get_shape()[-1]
        return tf.reshape( tf.reduce_mean( x, axis = [ 1, 2 ] ), ( -1, 1, 1, c ) )

def global_average_pool_channel(x):
    with tf.variable_scope('global_average_pool_channel'):
        h = x.get_shape()[1]
        w = x.get_shape()[2]
        return tf.reshape( tf.reduce_mean( x, axis = 3 ), ( -1, h, w, 1 ) )

def global_max_pool_spartial(x):
    with tf.variable_scope('global_max_pool_spartial'):
        c = x.get_shape()[-1]
        return tf.reshape( tf.reduce_max( x, axis = [ 1, 2 ] ), ( -1, 1, 1, c ) )

def global_max_pool_channel(x):
    with tf.variable_scope('global_max_pool_channel'):
        h = x.get_shape()[1]
        w = x.get_shape()[2]
        return tf.reshape( tf.reduce_max( x, axis = 3 ), ( -1, h, w, 1 ) )

def zero_padding2d(x, pad=(3, 3)):
    paddings = tf.constant([[0, 0], [pad[0], pad[0]],
                            [pad[1], pad[1]], [0, 0]])
    return tf.pad(x, paddings, 'CONSTANT')

# =============================================== non-lieanr functions =============================================== 

"""
From OpenAI
"""  
def gelu(x, name=""):
    with tf.variable_scope('gelu'):
        x = 0.5 * x * ( 1 + tf.tanh( np.sqrt( 2 / np.pi ) * ( x + 0.044715 * tf.pow( x, 3 ) ) ) )
        # print(x)
        return x

def sigmoid(x, name=""):
    with tf.variable_scope('sigmoid'):
        return tf.nn.sigmoid( x, name = name )

def relu(x, name=""):
    with tf.variable_scope('relu'):
        return tf.nn.relu( x, name = name )

def softmax(x, axis=-1):
    with tf.variable_scope('softmax'):
        x = x - tf.reduce_max( x, axis = axis, keepdims = True )
        ex = tf.exp(x)
        return ex / tf.reduce_sum( ex, axis = axis, keepdims = True )

def lrelu(x, leak=0.2, name=""):
    with tf.variable_scope('lrelu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def prelu(x, name=''):
    with tf.variable_scope('prelu'):
        alphas = tf.get_variable( '{}_alpha'.format( name ), x.get_shape()[-1],
                                initializer = tf.constant_initializer(0.0),
                                dtype = tf.float32)
        pos = tf.nn.relu( x )
        neg = alphas * ( x - abs( x ) ) * 0.5
        x = tf.add( pos, neg )
        # print( x )
        return x

# =============================================== norm =============================================== 

def bn(x, center=True, scale=True, decay=0.9, is_training=False, name=''):
    with tf.variable_scope('batch_norm'):
        x = tf.layers.batch_normalization( x, trainable = is_training, training = is_training, center = center, scale = scale, name = name )
        return x

"""
From OpenAI
"""  
def norm(x, name, axis=-1, epsilon=1e-5, is_training=False):

    n_state = x.shape[ axis ].value
    shape = x.shape.as_list()
    shape[axis] = 1
    
    b = tf.compat.v1.get_variable( '{}_norm_b'.format( name ), [ n_state ], initializer = tf.constant_initializer(0), trainable = is_training )
    g = tf.compat.v1.get_variable( '{}_norm_g'.format( name ), [ n_state ], initializer = tf.constant_initializer(1), trainable = is_training )
    # mean = tf.compat.v1.get_variable( '{}_moving_mean'.format( name ), shape[1:], initializer = tf.zeros_initializer(), trainable = False )
    # variance = tf.compat.v1.get_variable( '{}_moving_var'.format( name ), shape[1:], initializer = tf.zeros_initializer(), trainable = False )
    
    # if is_training:

    u = tf.reduce_mean( x, axis = axis, keepdims = True)
    s = tf.reduce_mean( tf.square( x - u ), axis = axis, keepdims = True )

    x = ( x - u ) * tf.rsqrt( s + epsilon )
    x = x * g + b        

        # update_mean = tf.assign( mean, mean * 0.9 + 0.1 * u )
        # update_variance = tf.assign( variance, variance * 0.9 + 0.1 * s )

        # tf.add_to_collection( update_mean, tf.GraphKeys.UPDATE_OPS )
        # tf.add_to_collection( update_variance, tf.GraphKeys.UPDATE_OPS )
    
    # else:
    #     x = ( x - mean ) / tf.rsqrt( variance + epsilon )
    
    return x

def spectral_norm(w, name, iteration=1):
    
    w_shape = w.shape.as_list()
    w = tf.reshape( w, [ -1, w_shape[-1] ] )

    u_var = tf.compat.v1.get_variable( "{}_u".format( name ), [ 1, w_shape[-1]],
                                       initializer = tf.truncated_normal_initializer(),
                                       trainable = False )

    u = u_var

    for _ in range(iteration):
        v = tf.nn.l2_normalize( tf.matmul( u, w, transpose_b = True ) )
        u = tf.nn.l2_normalize( tf.matmul( v, w ) )

    sigma = tf.squeeze( tf.matmul( tf.matmul( v, w ), u, transpose_b = True ) )
    w = tf.div_no_nan( w, sigma )

    with tf.control_dependencies([ u_var.assign( u ) ]):
        w = tf.reshape( w, w_shape )

    return w

# ==============================================================================================

def noise(x, _mu=0.5, theta=0.15, sigma=0.2, stdev=1.0):
    mu = _mu * tf.ones( tf.shape( x ) )
    random_pos = theta * ( mu - x ) + sigma * tf.random.normal( tf.shape( x ), stddev = stdev )
    return random_pos

def get_median(v):
    v = tf.reshape(v, [-1])
    mid = v.get_shape()[0]//2 + 1
    return tf.nn.top_k(v, mid).values[-1]

"""
From OpenAI
"""  
def positional_encoding(x, position, d_model):

    def get_angles(position, i, d_model):
        angles = 1 / tf.pow( 10000, ( 2 * ( i // 2 ) ) / tf.cast( d_model, tf.float32 ) )
        return position * angles

    angle_rads = get_angles( 
        position = tf.range( position, dtype = tf.float32 )[:, tf.newaxis],
        i = tf.range( d_model, dtype = tf.float32 )[tf.newaxis, :],
        d_model = d_model )
    
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    
    return x + tf.cast(pos_encoding, tf.float32)[:, :tf.shape(x)[1], :]