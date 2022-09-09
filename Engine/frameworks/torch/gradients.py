import torch


def eval_and_grad(net, x, target, loss_fn):
    x.requires_grad = True
    y = net(x)
    loss = loss_fn(y, target)

    loss.retain_grad()
    loss.backward()
    return loss.detach().numpy(), [p.grad for p in net.params()]
