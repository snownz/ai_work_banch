import tensorflow as tf
import numpy as np

from ann_utils.helper import flatten, maxpool2d
from ann_utils.som_layer import SOMLayer

class StackedMemoryBlock(object):

    def __init__( self,
                  name,
                  m, n,
                  epoch,
                  lr,
                  nr,
                  gstd,
                  stack,
                  act=None,
                  polling=None
                ):

        self.blocks = [ MemoryBlock( "{}_{}_mem_block".format( i, name ), 
                        m, n, epoch, lr, nr, gstd, act, polling )                         
                        for i in range( stack ) ]

    def __call__(self, x):
        mems = None
        for m in self.blocks:
            mi = m( x, is_training = False )
            fmi = flatten( mi )
            if mems is None:
                mems = mi
            else:
                mems = tf.concat( [ mems, mi ], axis = 3 )
            x = fmi        
        return mems

    def update_memory(self, x):
        update = []
        for m in self.blocks:
            mi, up = m( x, is_training = True )
            fmi = flatten( mi )
            x = fmi
            update.extend( up ) 
        return update

class MemoryBlock(object):

    def __init__( self,
                  name,
                  m, n,
                  epoch,
                  lr,
                  nr,
                  gstd,
                  act=None,
                  polling=None
                ):

        self.name = name 
        self.m = m 
        self.n = n 
        self.epoch = epoch
        self.lr = lr 
        self.nr = nr
        self.act = act
        self.polling = polling

        self.blockm = SOMLayer( "{}_block_m".format( name ), m, n, epoch, lr, nr, gstd, act )

    def __call__(self, x, is_training=False):
        m, mu = self.blockm( x, is_training = is_training )                
        print(m)               
        if is_training:
            return m, mu
        return m