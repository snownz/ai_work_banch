import os, gc, threading, logging, time
import tensorflow as tf
import cv2 as cv
import numpy as np

from random import randint
from time import sleep

# ambiente
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY
from warpers.gym_warper import wrapper

# modelos
from ann_utils.agents.curiosity_driven_agent import Curiosity_DDQN_Context_Agent, DDQN
from ann_utils.models.specialists.vae_transformer_model import LSTM_Curiosity_Memory_DQN

# auxiliares
from ann_utils.manager import tf_global_initializer, tf_load, tf_save
from ann_utils.sess import TfSess

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto( log_device_placement = False, 
                         allow_soft_placement = True ) 
config.gpu_options.allow_growth = True
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.gpu_options.per_process_gpu_memory_fraction = 0.9

sess = TfSess( 'mario', gpu = True, config = config )

act_s = len( COMPLEX_MOVEMENT )
chp = "./saved/mario/"

bs = 128
s_s = 128
i_s = [ None, None, 3 ]

writer = tf.summary.FileWriter( './tmp/tensorflow/mario/ddqn' )
model = LSTM_Curiosity_Memory_DQN( ( 96, 96 ), act_s, s_s )
agent = Curiosity_DDQN_Context_Agent( model, 1000, bs, 100000 ).\
build( i_s, s_s, act_s, sess, False, False )

# genv = gym_super_mario_bros.make( 'SuperMarioBrosRandomStages-v0' )
# genv = JoypadSpace( genv, RIGHT_ONLY )
# genv = wrapper( genv )

# genv = JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v2'), RIGHT_ONLY )
genv = JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v2'), COMPLEX_MOVEMENT )

agents_controller = DDQN( genv, agent, writer, num_local_steps = bs )

init_op = tf.initialize_all_variables()
sess( init_op )

writer.add_graph( sess.get_session().graph )

agents_controller.start()
# variables = tf.global_variables()
