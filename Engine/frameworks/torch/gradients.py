import torch


def eval_and_grad(net, x, target, loss_fn):
    x.requires_grad = True
    y = net(x)
    loss = loss_fn(y, target)

    loss.retain_grad()
    loss.backward()
    return loss.detach().numpy(), [p.grad for p in net.parameters()]


def gradient_step(net, grads):
    for epoch in range(10):
        for p, g in zip(net.parameters(), grads):
            p = p.detach()
            p -= 0.001 * g
            g = None

