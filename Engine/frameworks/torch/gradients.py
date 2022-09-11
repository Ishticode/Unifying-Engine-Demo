def eval_and_grad(net, x, target, loss_fn):

    x.requires_grad = True
    y = net(x)
    loss = loss_fn(y, target)

    loss.retain_grad()
    loss.backward()
    return loss.item(), [p.grad for p in net.parameters()]


def gradient_step(net, grads):

    for p, g in zip(net.parameters(), grads):
        p = p.detach()
        p -= 0.0001 * g


