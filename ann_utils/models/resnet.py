import sys
sys.path.append('../')

from ann_utils.residual_block import *
from ann_utils.conv_layer import Conv2DLayer
from ann_utils.helper import avgpool2d, zero_padding2d, maxpool2d

class Resnet50(object):

    def __init__( self,
                  name,
                  kernel,
                  dropout = 0.0, bn = False, ln = False,
                  act = tf.nn.leaky_relu ):

        # stage 1
        self.st1_c = Conv2DLayer( 64, kernel[0], 1, '{}_st1_c'.format( name ), 
                                  dropout, bn, ln, "VALID", act )

        # stage 2
        self.st2_b  = ConvBlock( "{}_st2_b".format( name ),      kernel[1], [64, 64, 256], 1, bn = bn, ln = ln )
        self.st2_i1 = IdentityBlock( "{}_st2_i1".format( name ), kernel[1], [64, 64, 256], bn = bn, ln = ln )
        self.st2_i2 = IdentityBlock( "{}_st2_i2".format( name ), kernel[1], [64, 64, 256], bn = bn, ln = ln )

        # stage 3
        self.st3_b  = ConvBlock( "{}_st3_b".format( name ),      kernel[2], [128, 128, 512], 2, bn = bn, ln = ln )
        self.st3_i1 = IdentityBlock( "{}_st3_i1".format( name ), kernel[2], [128, 128, 512], bn = bn, ln = ln )
        self.st3_i2 = IdentityBlock( "{}_st3_i2".format( name ), kernel[2], [128, 128, 512], bn = bn, ln = ln )
        self.st3_i3 = IdentityBlock( "{}_st3_i3".format( name ), kernel[2], [128, 128, 512], bn = bn, ln = ln )

        # stage 4
        self.st4_b  = ConvBlock( "{}_st4_b".format( name ),      kernel[3], [256, 256, 1024], 2, bn = bn, ln = ln )
        self.st4_i1 = IdentityBlock( "{}_st4_i1".format( name ), kernel[3], [256, 256, 1024], bn = bn, ln = ln )
        self.st4_i2 = IdentityBlock( "{}_st4_i2".format( name ), kernel[3], [256, 256, 1024], bn = bn, ln = ln )
        self.st4_i3 = IdentityBlock( "{}_st4_i3".format( name ), kernel[3], [256, 256, 1024], bn = bn, ln = ln )
        self.st4_i4 = IdentityBlock( "{}_st4_i4".format( name ), kernel[3], [256, 256, 1024], bn = bn, ln = ln )
        self.st4_i5 = IdentityBlock( "{}_st4_i5".format( name ), kernel[3], [256, 256, 1024], bn = bn, ln = ln )

        # stage 5
        self.st5_b  = ConvBlock( "{}_st5_b".format( name ),      kernel[4], [512, 512, 2048], 2, bn = bn, ln = ln )
        self.st5_i1 = IdentityBlock( "{}_st5_i1".format( name ), kernel[4], [512, 512, 2048], bn = bn, ln = ln )
        self.st5_i2 = IdentityBlock( "{}_st5_i2".format( name ), kernel[4], [512, 512, 2048], bn = bn, ln = ln )
        

    def __call__(self, x, is_training=False): 

        # stage 1
        x = zero_padding2d( x, ( 4, 4 ) )
        x = self.st1_c( x, is_training )
        x1 = maxpool2d( x, 2, 2, padding='VALID' )

        # stage 2
        x = self.st2_b( x1, is_training )
        x = self.st2_i1( x, is_training )
        x2 = self.st2_i2( x, is_training )

        # stage 3
        x = self.st3_b( x2, is_training )
        x = self.st3_i1( x, is_training )
        x = self.st3_i2( x, is_training )
        x3 = self.st3_i3( x, is_training )

        # stage 4
        x = self.st4_b( x3, is_training )
        x = self.st4_i1( x, is_training )
        x = self.st4_i2( x, is_training )
        x = self.st4_i3( x, is_training )
        x = self.st4_i4( x, is_training )
        x4 = self.st4_i5( x, is_training )

        # stage 5
        x = self.st5_b( x4, is_training )
        x = self.st5_i1( x, is_training )
        x5 = self.st5_i2( x, is_training )

        return x1, x2, x3, x4, x5


class Resnet101(object):

    def __init__( self,
                  name,
                  dropout = 0.0, bn = False,
                  act = tf.nn.leaky_relu ):

        # stage 1
        self.st1_c = Conv2DLayer( 64, 7, 2, '{}_st1_c'.format( name ), 
                               dropout, bn, "VALID", act )

        # stage 2
        self.st2_b  = ConvBlock( "{}_st2_b".format( name ),      3, [64, 64, 256], 1 )
        self.st2_i1 = IdentityBlock( "{}_st2_i1".format( name ), 3, [64, 64, 256] )
        self.st2_i2 = IdentityBlock( "{}_st2_i2".format( name ), 3, [64, 64, 256] )

        # stage 3
        self.st3_b  = ConvBlock( "{}_st3_b".format( name ),      3, [128, 128, 512], 2 )
        self.st3_i1 = IdentityBlock( "{}_st3_i1".format( name ), 3, [128, 128, 512] )
        self.st3_i2 = IdentityBlock( "{}_st3_i2".format( name ), 3, [128, 128, 512] )
        self.st3_i3 = IdentityBlock( "{}_st3_i3".format( name ), 3, [128, 128, 512] )

        # stage 4
        self.st4_b  = ConvBlock( "{}_st4_b".format( name ),      3, [256, 256, 1024], 2 )
        self.st4_i1 = IdentityBlock( "{}_st4_i1".format( name ), 3, [256, 256, 1024] )
        self.st4_i2 = IdentityBlock( "{}_st4_i2".format( name ), 3, [256, 256, 1024] )
        self.st4_i3 = IdentityBlock( "{}_st4_i3".format( name ), 3, [256, 256, 1024] )
        self.st4_i4 = IdentityBlock( "{}_st4_i4".format( name ), 3, [256, 256, 1024] )
        self.st4_i5 = IdentityBlock( "{}_st4_i5".format( name ), 3, [256, 256, 1024] )
        self.st4_i6 = IdentityBlock( "{}_st4_i6".format( name ), 3, [256, 256, 1024] )
        self.st4_i7 = IdentityBlock( "{}_st4_i7".format( name ), 3, [256, 256, 1024] )
        self.st4_i8 = IdentityBlock( "{}_st4_i8".format( name ), 3, [256, 256, 1024] )
        self.st4_i9 = IdentityBlock( "{}_st4_i9".format( name ), 3, [256, 256, 1024] )
        self.st4_i10 = IdentityBlock( "{}_st4_i10".format( name ), 3, [256, 256, 1024] )
        self.st4_i11 = IdentityBlock( "{}_st4_i11".format( name ), 3, [256, 256, 1024] )
        self.st4_i12 = IdentityBlock( "{}_st4_i12".format( name ), 3, [256, 256, 1024] )
        self.st4_i13 = IdentityBlock( "{}_st4_i13".format( name ), 3, [256, 256, 1024] )
        self.st4_i14 = IdentityBlock( "{}_st4_i14".format( name ), 3, [256, 256, 1024] )
        self.st4_i15 = IdentityBlock( "{}_st4_i15".format( name ), 3, [256, 256, 1024] )
        self.st4_i16 = IdentityBlock( "{}_st4_i16".format( name ), 3, [256, 256, 1024] )
        self.st4_i17 = IdentityBlock( "{}_st4_i17".format( name ), 3, [256, 256, 1024] )
        self.st4_i18 = IdentityBlock( "{}_st4_i18".format( name ), 3, [256, 256, 1024] )
        self.st4_i19 = IdentityBlock( "{}_st4_i19".format( name ), 3, [256, 256, 1024] )
        self.st4_i20 = IdentityBlock( "{}_st4_i20".format( name ), 3, [256, 256, 1024] )
        self.st4_i21 = IdentityBlock( "{}_st4_i21".format( name ), 3, [256, 256, 1024] )
        self.st4_i22 = IdentityBlock( "{}_st4_i22".format( name ), 3, [256, 256, 1024] )

        # stage 5
        self.st5_b  = ConvBlock( "{}_st5_b".format( name ),      3, [512, 512, 2048], 2 )
        self.st5_i1 = IdentityBlock( "{}_st5_i1".format( name ), 3, [512, 512, 2048] )
        self.st5_i2 = IdentityBlock( "{}_st5_i2".format( name ), 3, [512, 512, 2048] )
        

    def __call__(self, x, is_training=False): 

        # stage 1
        x = zero_padding2d( x, ( 3, 3 ) )
        x = self.st1_c( x, is_training )
        x =  maxpool2d( x, 3, 2, padding='VALID' )

        # stage 2
        x = self.st2_b( x, is_training )
        x = self.st2_i1( x, is_training )
        x = self.st2_i2( x, is_training )

        # stage 3
        x = self.st3_b( x, is_training )
        x = self.st3_i1( x, is_training )
        x = self.st3_i2( x, is_training )
        x = self.st3_i3( x, is_training )

        # stage 4
        x = self.st4_b( x, is_training )
        x = self.st4_i1( x, is_training )
        x = self.st4_i2( x, is_training )
        x = self.st4_i3( x, is_training )
        x = self.st4_i4( x, is_training )
        x = self.st4_i5( x, is_training )
        x = self.st4_i6( x, is_training )
        x = self.st4_i7( x, is_training )
        x = self.st4_i8( x, is_training )
        x = self.st4_i9( x, is_training )
        x = self.st4_i10( x, is_training )
        x = self.st4_i11( x, is_training )
        x = self.st4_i12( x, is_training )
        x = self.st4_i13( x, is_training )
        x = self.st4_i14( x, is_training )
        x = self.st4_i15( x, is_training )
        x = self.st4_i16( x, is_training )
        x = self.st4_i17( x, is_training )
        x = self.st4_i18( x, is_training )
        x = self.st4_i19( x, is_training )
        x = self.st4_i20( x, is_training )
        x = self.st4_i21( x, is_training )
        x = self.st4_i22( x, is_training )
        
        # stage 5
        x = self.st5_b( x, is_training )
        x = self.st5_i1( x, is_training )
        x = self.st5_i2( x, is_training )

        x = avgpool2d( x, 2, 2 , padding='VALID' )

        return x

        