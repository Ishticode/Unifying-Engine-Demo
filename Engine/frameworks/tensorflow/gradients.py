import tensorflow as tf


def eval_and_grad(net, x, target, loss_fn):
    x = tf.Variable(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        tape.watch([p for p in net.parameters()])

        y = net(x)
        loss = loss_fn(y, target)
        grads = tape.gradient(loss, [p for p in net.parameters()])

    return loss.numpy(), grads


def gradient_step(net, grads):
    new_params = []
    for p, g in zip(net.parameters(), grads):
        p = tf.Variable(tf.stop_gradient(p))
        p.assign(tf.subtract(p, 0.001 * g))
        new_params.append(p)
    [p.assign(n) for p, n in zip(net.parameters(), new_params)]
