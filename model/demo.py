import tensorflow as tf
import numpy as np

identity = tf.identity
def conv_maxpool_layer(x, name, filters, kernel_size, padding='same', \
        conv_stride=1, pool_stride=2, pool_size=[2, 2],
        activation=tf.nn.relu):

    with tf.name_scope(name):
        x = tf.layers.conv2d(inputs=x, filters=filters, \
                kernel_size=kernel_size, strides=conv_stride, \
                padding=padding, activation=activation, \
                name='%s.conv'%name)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, 
                strides=pool_stride, name='%s.maxpool'%name)
        return x

def prod(x):
    #   return production of an array
    return int(np.array(x).prod())

def dense_layer(x, is_training, name='dense_layer', \
        units=1024, activation=tf.nn.relu):
    with tf.name_scope('dense_layer'):
        #   reshape tensor from [batch_size, w, h, c] 
        #   to [batch_size, w*h*c]
        x = tf.reshape(x, (-1, prod(x.shape[1:])), name='%s.reshape'%name) 
        x = tf.layers.dense(inputs=x, units=units, activation=activation, \
                name='%s.fully_connect'%name)
        x = tf.layers.dropout(inputs=x, rate=0.4, \
                training=is_training,   # only works when training
                name='%s.dropout'%name)
        return x

def demo_model(x, is_training):

    params = [
        {'filters': 32, 'kernel_size': [5, 5]},
        {'filters': 64, 'kernel_size': [5, 5]},
    ]
    
    #   two layers with conv & maxpool
    for i in range(len(params)):
        x = conv_maxpool_layer(x, 'layer_%d'%i, **params[i])
    
    #   flat & fully connect layer & dropout
    x = dense_layer(x, is_training, 'dense_layer')
    return x


       
