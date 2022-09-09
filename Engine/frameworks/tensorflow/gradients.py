import tensorflow as tf


def eval_and_grad(net, x, target, loss_fn):

    with tf.GradientTape() as tape:
        tape.watch(x)
        tape.watch(net.params())

        y = net(x)
        loss = loss_fn(y, target)
        grads = tape.gradient(loss, net.params())

    return loss, grads
