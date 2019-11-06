import sys
sys.path.append('../')

import tensorflow as tf

from ann_utils.models.squeeze_n_excitation import SqueezeNExcitation
from ann_utils.dense_block import DesneBlock, TransitionBlock
from ann_utils.conv_layer import Conv2DLayer
from ann_utils.helper import zero_padding2d, maxpool2d, bn, ln

class DenseNet121(object):

    def __init__( self,
                  name,
                  nb_dense_block=4, growth_rate=32, reduction=0.0,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        # compute compression factor
        compression = 1.0 - reduction

        self.act = act
        self.nb_dense_block = nb_dense_block
        nb_filter = 32
        nb_layers = [6,12,24,16] # For DenseNet-121

        self.c1 = Conv2DLayer( nb_filter, 7, 2, "{}_c1".format( name ), act = None, bias = False )

        self.d_blocks = []
        self.transition_block = []
        self.sne = []
        for block_idx in range( self.nb_dense_block - 1 ):
            b = DesneBlock( "{}_{}_dcb".format( name, block_idx ), nb_layers[block_idx], nb_filter, growth_rate, dropout = dropout )
            nb_filter = b.nb_filter
            t = TransitionBlock( "{}_{}_tb".format( name, block_idx ), compression = compression, dropout = dropout )
            nb_filter = int( nb_filter * compression )
            self.d_blocks.append( b )
            self.transition_block.append( t )
            self.sne.append( SqueezeNExcitation( "{}_{}_sne".format( name, block_idx ), 4 ) )
        
        self.d_blocks.append( DesneBlock( "{}_{}_dcb".format( name, len( self.d_blocks ) ), nb_layers[-1], nb_filter, growth_rate, dropout = dropout ) )
        self.sne.append( SqueezeNExcitation( "{}_{}_sne".format( name, len( self.d_blocks ) ), 16 ) )

    def __call__(self, x, is_training=False): 

        # Initial convolution
        x = self.c1( x, is_training )
        x = bn( x, is_training )
        x = self.act( x )
        x = maxpool2d( x, 3, 2 )

        # Add dense blocks
        for block_idx in range( self.nb_dense_block - 1 ):            
            x = self.d_blocks[block_idx]( x, is_training )
            # Add transition_block
            x = self.transition_block[block_idx]( x, is_training )
            x = self.sne[block_idx]( x, is_training )
        x = self.d_blocks[-1]( x, is_training )
        self.sne[-1]( x, is_training )

        x = bn( x, is_training )
        x = self.act( x )

        return x