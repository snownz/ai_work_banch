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
from ann_utils.agents.curiosity_driven_agent import Curiosity_AC_Context_Agent, A2C
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
devices = sess.get_session().list_devices()

action_size = len( COMPLEX_MOVEMENT )
chp = "./saved/mario/"

bs = 16
state_size = 128

model = LSTM_Curiosity_Memory_AC( ( 48, 48 ), action_size, state_size )
agent = Curiosity_AC_Context_Agent( model, 1000, bs, 100000 )

global_writer = tf.summary.FileWriter( './tmp/tensorflow/mario/a2c/global2' )

global_agent = agent.build_agent_brain(
    0.0, [ None, None, 3 ], state_size, action_size,
    devices[2], devices[2], sess,
    True, 'mario_global', True, None )

genv = JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0'), COMPLEX_MOVEMENT )

agents_controller = A2C( genv, global_agent, global_writer, curiosity = True, num_local_steps = bs )

tf_global_initializer( sess )

agents_controller.start()
variables = tf.global_variables()

while True:
    sleep(2.4)
    print("Main Thread: Optmizing")
    agents_controller.process( sess )
#    tf_save( chp, variables, "mario", sess, True )

# variables = tf.global_variables()
# tf_save( chp, variables, "mario", sess, True )

