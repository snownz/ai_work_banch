import os, gc
import tensorflow as tf
import numpy as np

def tf_reset_graph(session):        
    tf.reset_default_graph()
    gc.collect()    
    session.reset()    

def tf_parcial_initializer(session):
    for x in tf.global_variables():
        if not session( tf.is_variable_initialized( x ) ):
            session( x.initializer )

def tf_global_initializer(session):
    session( tf.compat.v1.global_variables_initializer() )

def tf_save(folder, params, name, sess, lst=False):
    if not os.path.isdir( folder ):
        os.makedirs( folder )   
    if lst:
        saver = tf.train.Saver( params )
    else:
        saver = tf.train.Saver( params[ 1 ] )
    saver.save( sess.get_session(), "{}{}".format( folder, name ) )
        
def tf_load(folder, params, name, sess, lst=False):
    if not os.path.isdir( folder ):
        return None
    print("Folder: {}".format( folder ) )
    if lst:
        saver = tf.train.Saver( params )
    else:
        saver = tf.train.Saver( params[ 1 ] )
    saver.restore( sess.get_session(), "{}{}".format( folder, name ) )