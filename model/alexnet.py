import tensorflow as tf
import numpy as np

identity = tf.identity
def conv_maxpool_lrn_layer(x, name, filters, kernel_size, padding='valid', \
        conv_stride=1, pool_stride=2, pool_size=2, activation=tf.nn.relu):

    with tf.name_scope(name):
        x = tf.layers.conv2d(inputs=x, filters=filters, \
                kernel_size=kernel_size, strides=conv_stride, \
                padding=padding, activation=activation, \
                name='%s.conv'%name)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, 
                strides=pool_stride, name='%s.maxpool'%name)
        x = tf.nn.local_response_normalization(x)
        return x

def prod(x):
    #   return production of an array
    return int(np.array(x).prod())

def dense_layer(x, is_training, name='dense_layer', \
        units=4096, activation=tf.nn.tanh):
    with tf.name_scope('dense_layer'):
        #   reshape tensor from [batch_size, w, h, c] 
        #   to [batch_size, w*h*c]
        x = tf.reshape(x, (-1, prod(x.shape[1:])), name='%s.reshape'%name) 
        for i in range(2):  
            x = tf.layers.dense(inputs=x, units=units, \
                    activation=activation,  \
                    name='%s.fully_connect_%d'%(name, i))
            x = tf.layers.dropout(inputs=x, rate=0.5, \
                    training=is_training,   # only works when training
                    name='%s.dropout_%d'%(name, i))
        return x

def alexnet_model(x, is_training):

    params = [
        {'filters': 96, 'kernel_size': 11, 'conv_stride': 4, \
                'pool_stride': 2, 'pool_size': 3},
        {'filters': 256, 'kernel_size': 5, 'conv_stride': 1, \
                'pool_stride': 2, 'pool_size': 3},
    ]
    
    #   two layers with conv & maxpool
    for i in range(len(params)):
        x = conv_maxpool_lrn_layer(x, 'layer_%d'%i, **params[i])

    x = tf.layers.conv2d(inputs=x, filters=384, kernel_size=3, \
            activation=tf.nn.relu, name='conv_2')
    x = tf.layers.conv2d(inputs=x, filters=384, kernel_size=3, \
            activation=tf.nn.relu, name='conv_3')

    param_4 = {'filters': 256, 'kernel_size': 3, 'conv_stride': 1, \
                'pool_stride': 2, 'pool_size': 3}
    x = conv_maxpool_lrn_layer(x, 'layer_4', **param_4)
    
    #   flat & fully connect layer & dropout
    x = dense_layer(x, is_training, 'dense_layer')
    return x


       
