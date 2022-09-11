import tensorflow as tf


def linear(x, weight, bias=None):
    return (tf.matmul(x, tf.transpose(weight)) + bias) if bias is not None else tf.matmul(x, tf.transpose(weight))


def conv2d(x, filters, strides, padding, dilations=1):
    res = tf.nn.conv2d(x, filters, strides, padding, 'NHWC', dilations)
    return res


def relu(x):
    return tf.nn.relu(x)


def max_pool(x, ksize, strides):
    res = tf.nn.max_pool(x, ksize, strides, 'VALID', 'NHWC')
    return res


def mse_loss(x, y):
    return tf.reduce_mean(tf.square(x - y))
