import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import common

tf.compat.v1.disable_eager_execution()  #<--- Disable eager execution

def inference(x, depth, num_output, wd, is_training, transfer_mode= False):
    num_blockes= []
    bottleneck= False
    if depth == 18:
      num_blocks= [2, 2, 2, 2]
    elif depth == 34:
      num_blocks= [3, 4, 6, 3]
    elif depth == 50:
      num_blocks= [3, 4, 6, 3]
      bottleneck= True
    elif depth == 101:
      num_blocks= [3, 4, 23, 3]
      bottleneck= True
    elif depth == 152:
      num_blocks= [3, 8, 36, 3]
      bottleneck= True

    return getModel(x, num_output, wd, is_training, num_blocks= num_blocks, bottleneck= bottleneck, transfer_mode= transfer_mode)


def getModel(x, num_output, wd, is_training, num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              bottleneck= True, transfer_mode= False):
    conv_weight_initializer = tf.compat.v1.truncated_normal_initializer(stddev= 0.1)
    fc_weight_initializer = tf.compat.v1.truncated_normal_initializer(stddev= 0.01)
    with tf.compat.v1.variable_scope('scale1'):
        x = common.spatialConvolution(x, 7, 2, 64, weight_initializer= conv_weight_initializer, wd= wd)
        x = common.batchNormalization(x, is_training= is_training)
        x = tf.nn.relu(x)

    with tf.compat.v1.variable_scope('scale2'):
        x = common.maxPool(x, 3, 2)
        x = common.resnetStack(x, num_blocks[0], 1, 64, bottleneck, wd= wd, is_training= is_training)

    with tf.compat.v1.variable_scope('scale3'):
        x = common.resnetStack(x, num_blocks[1], 2, 128, bottleneck, wd= wd, is_training= is_training)

    with tf.compat.v1.variable_scope('scale4'):
        x = common.resnetStack(x, num_blocks[2], 2, 256, bottleneck, wd= wd, is_training= is_training)

    with tf.compat.v1.variable_scope('scale5'):
        x = common.resnetStack(x, num_blocks[3], 2, 512, bottleneck, wd= wd, is_training= is_training)

    # post-net
    x = tf.reduce_mean(input_tensor=x, axis= [1, 2], name= "avg_pool")
    output = [None]*len(num_output)
    for o in range(0,len(num_output)):
      with tf.compat.v1.variable_scope('output'+str(o)):
        output[o] = common.fullyConnected(x, num_output[o], weight_initializer= fc_weight_initializer, bias_initializer= tf.compat.v1.zeros_initializer, wd= wd)

    return output