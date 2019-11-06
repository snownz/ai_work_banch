import os, shutil
import tensorflow as tf
import uuid

from sklearn.exceptions import DataConversionWarning
from ann_utils.manager import tf_save
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils import build_signature_def

class TfSess(object):

    def __init__(self, name, percent=False, remote=None, gpu=False, config=None, folder='/tmp'):

        self.percent = percent
        self.name = name
        self.remote = remote
        self.gpu = gpu
        
        if os.path.isdir( '{}/tensorflow/{}'.format( folder, name ) ):
            name += '_' + uuid.uuid4().hex

        if isinstance( name, list ):
            self.writer = [ tf.compat.v1.summary.FileWriter( '{}/tensorflow/{}'.format( folder, x ) ) for x in name ]
            self.merged = []
        else:
            self.writer = tf.compat.v1.summary.FileWriter( '{}/tensorflow/{}'.format( folder, name ) )
        
        self.session = None
        self.config = config
        self.reset()        

    def reset(self):

        if not self.session is None:
            self.session.close()

        if not self.gpu:
            config = tf.compat.v1.ConfigProto( device_count = { 'GPU': 0 } )
        else:
            config = tf.compat.v1.ConfigProto( device_count = { 'GPU': 1 } )
                    
        if self.remote is None:
            self.session = tf.Session( config = self.config )
        else:
            self.session = tf.Session( self.remote, config = self.config )

        if self.gpu:
            device_name = tf.test.gpu_device_name()
            print('Found GPU at: {}'.format(device_name))

    def tensorboard_graph(self, index=0, scope=None):
        if isinstance( self.writer, list ):
            self.writer[ index ].add_graph( self.session.graph )
        else:
            self.writer.add_graph( self.session.graph )
    
    def merge_summary(self, summaries=None):

        if isinstance( self.writer, list ):
            self.merged.append( tf.summary.merge( summaries ) )
        else:        
            self.merged = tf.summary.merge_all()

    def freeze_pb_graph(self, folder, i_tensors, o_tensors):

        if os.path.isdir( folder ):
            shutil.rmtree( folder )

        model_input = { x[0]: tf.saved_model.build_tensor_info(x[1]) for x in i_tensors }
        model_output = { x[0]: tf.saved_model.build_tensor_info(x[1]) for x in o_tensors }

        builder = tf.saved_model.builder.SavedModelBuilder( folder )
        signature_definition = build_signature_def( 
            inputs = model_input,
            outputs = model_output,
            method_name = signature_constants.PREDICT_METHOD_NAME
         )

        builder.add_meta_graph_and_variables(
                self.session, [tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        signature_definition
                })
        builder.save()
        
    def get_session(self):
        return self.session

    def add_summary(self, s, step):
        self.writer.add_summary( s, step )
        self.writer.flush()

    def __call__(self, tensor, inputs=None, summary=False, step=0, index=1):
        
        if summary:

            if not type(tensor) is list:
                tensor = [ tensor ]
            if not self.merged in tensor:
                if isinstance( self.writer, list ):
                    tensor.append( self.merged[index] )
                else:
                    tensor.append( self.merged )
                
        result = self.session.run( tensor, feed_dict = inputs )
        
        if summary:
            r = result[ 0: len( result ) -1 ]
            s = result[-1]
            if isinstance( self.writer, list ):
                self.writer[index].add_summary( s, step )
            else: 
                self.writer.add_summary( s, step )
            
            return r
        else:
            return result