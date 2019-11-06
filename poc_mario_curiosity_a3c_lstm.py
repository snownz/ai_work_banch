import os, gc, threading, logging, time
import tensorflow as tf
import cv2 as cv
import numpy as np

from random import randint
from time import sleep

# ambiente
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

# modelos
from ann_utils.agents.curiosity_driven_agent import Curiosity_AC_Context_Agent, A3C
from ann_utils.models.specialists.vae_transformer_model import LSTM_Curiosity_Memory_AC

# auxiliares
from ann_utils.manager import tf_global_initializer, tf_load, tf_save
from ann_utils.sess import TfSess

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


config = tf.ConfigProto( log_device_placement = False, 
                         inter_op_parallelism_threads = 8,
                         allow_soft_placement = True ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

sess = TfSess( 'mario', gpu = True, config = config )
dvc = sess.get_session().list_devices()

act_s = len( SIMPLE_MOVEMENT )
chp = "./saved/mario/"

num_worker = 4
bs = 32
s_s = 128
i_s = [ None, None, 3 ]

model = LSTM_Curiosity_Memory_AC( ( 48, 48 ), act_s, s_s )
agent = Curiosity_AC_Context_Agent( model, 1000, bs, 100000 )

global_writer = tf.summary.FileWriter( './tmp/tensorflow/mario/a3c_global_norm' )
worker_writer = [ tf.summary.FileWriter( './tmp/tensorflow/mario/a3c_worker_{}'.format( x ) ) for x in range( num_worker ) ]

global_agent = Curiosity_AC_Context_Agent( model, 1000, bs, 100000 ).build_agent_brain( i_s, s_s, act_s, dvc[0], dvc[2], sess, True, False, True, 'mario_global', False, None )

workers = [ Curiosity_AC_Context_Agent( model, 1000, bs, 100000 ).build_agent_brain( i_s, s_s, act_s, dvc[0], dvc[2], sess, False, False, False, 'mario_local_{}'.format(w), True, global_agent.model.variables )
            for w in range( num_worker ) ]

genv = JoypadSpace( gym_super_mario_bros.make('SuperMarioBros-v0'), SIMPLE_MOVEMENT )
envs = [ JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0'), SIMPLE_MOVEMENT ) for x in range( num_worker ) ]

agents_controller = A3C( genv, envs, global_agent, workers, global_writer, num_local_steps = bs )

tf_global_initializer( sess )

global_writer.add_graph( sess.get_session().graph )

# agents_controller.start( worker_writer, True )
# variables = tf.global_variables()

# while True:
    #tf_save( chp, variables, "mario", sess, True )
# variables = tf.global_variables()
# tf_save( chp, variables, "mario", sess, True )