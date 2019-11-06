import sys
sys.path.append('../')

import tensorflow as tf

from ann_utils.conv_layer import Conv2DLayer
from ann_utils.helper import avgpool2d


class RegionProposalNetwork(object):

    def __init__(self, name, anchor_stride, anchors_per_location):

         self.shared = Conv2DLayer( 512, 3, 1, "{}_rpn_conv_shared".format( name ), padding='same', act = tf.nn.relu )
         self.anchor = Conv2DLayer( 2 * anchors_per_location, 1, 1, "{}_rpn_class_raw".format( name ), padding='valid', act = None )
         self.bounding = Conv2DLayer( anchors_per_location * 4, 1, 1, "{}_rpn_bbox_pred".format( name ), padding='valid', act = None )

    def __call__(self, x, is_training=False):

        # Shared convolutional base of the RPN
        shared = self.shared( x, is_training )

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = self.anchor( shared, is_training )

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = tf.reshape( x, [ tf.shape( x )[0], -1, 2 ] )

        # Softmax on last dimension of BG/FG.
        rpn_probs = tf.nn.softmax( rpn_class_logits )

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = self.bounding( shared )

        # Reshape to [batch, anchors, 4]
        rpn_bbox = tf.reshape( x, [ tf.shape( x )[0], -1, 4] ) 

        return [ rpn_class_logits, rpn_probs, rpn_bbox ]