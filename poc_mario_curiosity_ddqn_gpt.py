import os
import tensorflow as tf
import cv2 as cv
import numpy as np

# ambiente
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

# modelos
from ann_utils.agents.curiosity_driven_agent import Curiosity_DQN_GPT_Agent
from ann_utils.models.specialists.vae_transformer_model import GPT2_Curiosity_DQN

# auxiliares
from ann_utils.manager import tf_global_initializer, tf_load, tf_save
from ann_utils.sess import TfSess

from tqdm import tqdm

NORMAL_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['A'],
    ['B'],
    ['left'],
    ['down'],
    ['up'],
]

config = tf.ConfigProto( log_device_placement = False ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = TfSess( "mario_ddqn_gpt", gpu = True, config = config )

env = gym_super_mario_bros.make('SuperMarioBros-v0' )
env = JoypadSpace(env, COMPLEX_MOVEMENT)

state_info = env.reset()
action_info = env.action_space.sample()
action_size = env.action_space.n
chp = "./saved/mario/"

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(action_size))

size = [ 192, 192, 3 ]
bs = 8
state_size = 256
sequence_size = 4

model = GPT2_Curiosity_DQN( action_size, state_size, 128, 3, 4 )
agent = Curiosity_DQN_GPT_Agent( sess, model, 1.0, 100, 1, 2000, bs, 0.9 )

with tf.compat.v1.variable_scope( 'mario', reuse = tf.AUTO_REUSE ):
    agent.build_agent_brain( [ None, sequence_size ] + size, sequence_size, action_size )

tf_global_initializer( sess )

variables = tf.global_variables()

# tf_load( chp, variables, "mario", sess, True )
t_steps = agent.memory_size // 2
n_episodes = 100000

state = cv.resize( env.reset(), ( size[0], size[1] ) ) / 255.0
agent.reset( sequence_size, size, action_size )
for episode in range(n_episodes):

    total_reward = 0
    for t in range(t_steps):

        action = agent.choose_action( state, action_size )
        next_state, reward, done, info = env.step(action)
        next_state = cv.resize( next_state, ( size[0], size[1] ) ) / 255.0
        total_reward += reward
        agent.step( state, action, reward, next_state )

        env.render()

        if done:
            life = 2
            state = cv.resize( env.reset(), ( size[0], size[1] ) ) / 255.0
            agent.reset( sequence_size, size, action_size )
            break
        
        state = next_state

    for t in range( t_steps // bs ):
        agent.learn()

    # tf_save( chp, variables, "mario", sess, True )
env.close()
