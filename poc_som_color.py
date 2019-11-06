import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

from ann_utils.som_layer import SOMLayer

from ann_utils.manager import tf_global_initializer
from ann_utils.sess import TfSess

#Training inputs for RGBcolors
colors = np.array([[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

color_names = ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

test_data = {
    "stage1":{
        "colors": [ [ 0., 0., 0. ], [ 255., 255., 255. ] ],
        "names": [ 'black', 'white' ],
    }
}

colors = np.array([[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])

color_names = ['black', 'blue', 'darkblue', 'skyblue',
    'greyblue', 'lilac', 'green', 'red',
    'cyan', 'violet', 'yellow', 'white',
    'darkgrey', 'mediumgrey', 'lightgrey']


# hyper parameter
dim  = 1000
num_epoch = 8
batch_size = 15

som = SOMLayer( "color", dim, dim, num_epoch, 0.5, 1.3, 0.08 )
sess = TfSess( "color", gpu = True )

# create the graph
x = tf.placeholder(shape=[None,3],dtype=tf.float32)

# graph
som_graph = som( x, True )

# initialize variables
tf_global_initializer( sess )

# start the training
for iter in range(num_epoch):
    for current_train_index in range(0,len(colors),batch_size):
        currren_train = colors[current_train_index:current_train_index+batch_size]
        sess( som_graph, { x : currren_train } )

# get the trained map and normalize
trained_map = sess( som.getmap() ).reshape( dim, dim, 3 )
trained_map[:,:,0] = ( trained_map[:,:,0] - trained_map[:,:,0].min() ) / ( trained_map[:,:,0].max() - trained_map[:,:,0].min() )
trained_map[:,:,1] = ( trained_map[:,:,1] - trained_map[:,:,1].min() ) / ( trained_map[:,:,1].max() - trained_map[:,:,1].min() )
trained_map[:,:,2] = ( trained_map[:,:,2] - trained_map[:,:,2].min() ) / ( trained_map[:,:,2].max() - trained_map[:,:,2].min() )

# get cluest vector
locations = sess( som.getlocation(), { x : test_data['stage1']['colors'] } )

plt.imshow(trained_map.astype(float))
for i, m in enumerate(locations):
    plt.text(m[1], m[0], test_data['stage1']['names'][i], ha='center', va='center',bbox=dict(facecolor='white', alpha=0.5,lw=0)) 
plt.axis('off')
plt.title('Color SOM')
plt.show()
plt.close('all')
