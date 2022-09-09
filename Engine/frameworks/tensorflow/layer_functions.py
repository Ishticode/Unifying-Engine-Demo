import tensorflow as tf


def linear(x, weight, bias=None):
    return (tf.matmul(x, tf.transpose(weight)) + bias) if bias is not None else tf.matmul(x, tf.transpose(weight))


def conv2d(x, filters, strides, padding, data_format='NHWC'):
    if data_format == 'NCHW':
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.conv2d(x, filters, strides, padding, 'NHWC')
    if data_format == 'NCHW':
        return tf.transpose(res, (0, 3, 1, 2))
    return res
