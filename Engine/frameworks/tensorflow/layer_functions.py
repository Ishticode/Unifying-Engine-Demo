import tensorflow as tf


def linear(x, weight, bias=None):
    return (tf.matmul(x, tf.transpose(weight)) + bias) if bias is not None else tf.matmul(x, tf.transpose(weight))


def conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1):
    if data_format == 'NCHW':
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.conv2d(x, filters, strides, padding, 'NHWC', dilations)
    if data_format == 'NCHW':
        return _tf.transpose(res, (0, 3, 1, 2))
    return res


def relu(x):
    return tf.nn.relu(x)


def max_pool(x, ksize, strides, data_format='NHWC'):
    if data_format == 'NCHW':
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.max_pool(x, ksize, strides, 'SAME')
    if data_format == 'NCHW':
        return tf.transpose(res, (0, 3, 1, 2))
    return res


def mse_loss(x, y):
    return tf.reduce_sum(tf.losses.mean_squared_error(x, y))
