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

config = tf.ConfigProto( device_count = { 'GPU': 1 }, 
                         log_device_placement = False, 
                         allow_soft_placement = True ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

sess = TfSess( 'mario', gpu = True, config = config )

act_s = len( RIGHT_ONLY )
chp = "./saved/mario/"

bs = 256
s_s = 1024
i_s = [ None, None, 3 ]

writer = tf.summary.FileWriter( './tmp/tensorflow/mario/ddqn/0' )
model = LSTM_Curiosity_Memory_DQN( ( 24, 24 ), act_s, s_s )
agent = Curiosity_DDQN_Context_Agent( model, 1024, bs, 999999 ).\
build( i_s, s_s, act_s, sess, False, False )

# genv = JoypadSpace( gym_super_mario_bros.make('SuperMarioBrosRandomStages-v2'), COMPLEX_MOVEMENT )
genv = JoypadSpace( gym_super_mario_bros.make('SuperMarioBros-v0'), RIGHT_ONLY )
agents_controller = DDQN( genv, agent, writer, num_local_steps = 1000, update_steps = 1000 )

init_op = tf.initialize_all_variables()
sess( init_op )

writer.add_graph( sess.get_session().graph )

agents_controller.start()

"""
Steps:
  1 - Simple + lstm
    1 - Dropout
    2 - l2
    3 - l1
    4 - Estocastico

Problems:
  s1: 
    local minimal, com capacidade de guardar sequencias das fases
    1 - Modelo não aprende, NAN ( Corrigido, com RElU )
 s2:

"""