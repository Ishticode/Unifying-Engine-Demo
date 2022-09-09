import tensorflow as tf


def tensor(x):
    return tf.constant(x)


def sum(x, axis=None):
    return tf.reduce_sum(x, axis=axis)


def randn(shape):
    return tf.random.normal(shape)


def variable(x):
    return tf.Variable(x)


def sqrt(x):
    return tf.sqrt(x)


def zeros_like(x):
    return tf.zeros_like(x)

def abs(x):
    return tf.abs(x)
