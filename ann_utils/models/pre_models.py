import sys
sys.path.append('../../')

import tensorflow as tf

from ann_utils.helper import gelu, prelu, softmax, sigmoid, lrelu

MARIO_CONV = [
    { 'type': 'conv', 'out': 32, 'k': 8, 's': 4, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
    { 'type': 'se', 'act': tf.nn.relu },
    { 'type': 'conv', 'out': 64, 'k': 4, 's': 2, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
    { 'type': 'se', 'act': tf.nn.relu },
    { 'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
    { 'type': 'se', 'act': tf.nn.relu }
]

MARIO_DCONV = [
    { 'type': 'dconv', 'out': 64, 'k': 3, 's': 4, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
    { 'type': 'se', 'act': tf.nn.relu },
    { 'type': 'dconv', 'out': 32, 'k': 4, 's': 2, 'dp': .25, 'bn': True, 'act': prelu, 'bias': False },
    { 'type': 'se', 'act': tf.nn.relu },
    { 'type': 'dconv', 'out': 3,  'k': 8, 's': 1, 'dp': .25, 'bn': True, 'act': None, 'bias': False },
]

MARIO_256_CONV = [
    { 'type': 'conv', 'out': 32, 'k': 3, 's': 2, 'dp': .25, 'bn': True, 'act': prelu, 'bias': True },
    { 'type': 'se', 'act': tf.nn.relu },
    { 'type': 'conv', 'out': 32, 'k': 3, 's': 2, 'dp': .25, 'bn': True, 'act': prelu, 'bias': True },
    { 'type': 'se', 'act': tf.nn.relu },
    { 'type': 'conv', 'out': 32, 'k': 3, 's': 2, 'dp': .0, 'bn': True, 'act': prelu, 'bias': True },
    { 'type': 'se', 'act': tf.nn.relu },
]

MARIO_256_DCONV = [
    { 'type': 'conv', 'out': 32, 'k': 3, 's': 1, 'dp': .25, 'bn': True, 'act': tf.nn.elu, 'bias': False },
    { 'type': 'conv', 'out': 32, 'k': 3, 's': 1, 'dp': .25, 'bn': True, 'act': tf.nn.elu, 'bias': False },
    { 'type': 'conv', 'out': 1,  'k': 3, 's': 1, 'dp': .0, 'bn': True, 'act': tf.nn.sigmoid, 'bias': False },
]

MARIO_SIMPLE_GS_CONV = [
    { 'type': 'conv', 'out': 16, 'k': 8, 's': 4, 'dp': .25, 'bn': False, 'act': tf.nn.elu, 'bias': False },
    { 'type': 'conv', 'out': 32, 'k': 4, 's': 2, 'dp': .25, 'bn': False, 'act': tf.nn.elu, 'bias': False }
]

MARIO_SIMPLE_GS_DCONV = [
    { 'type': 'dconv', 'out': 32, 'k': 3, 's': 4, 'dp': .25, 'bn': False, 'act': tf.nn.elu, 'bias': False },
    { 'type': 'dconv', 'out': 16, 'k': 3, 's': 2, 'dp': .25, 'bn': False, 'act': tf.nn.elu, 'bias': False },
    { 'type': 'dconv', 'out': 3,  'k': 3, 's': 1, 'dp': .25, 'bn': False, 'act': None, 'bias': False },
]

MARIO_ACT_REWARD_ENCODE = [
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 2, 'dp': .25, 'act': None, 'bias': False },
]

MARIO_ACT_REWARD_DECODE = [
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': 16, 'dp': .25, 'act': prelu, 'bias': False },
    { 'type': 'fully', 'size': None, 'dp': .25, 'act': None, 'bias': False },
]
