import torch


def tensor(x):
    return torch.tensor(x)


def sum(x, axis=None):
    if axis is None:
        return torch.sum(x)
    return torch.sum(x, dim=axis)


def randn(shape):
    return torch.randn(shape)


def variable(x):
    if isinstance(x, torch.Tensor):
        x.requires_grad = True
        return x
    return torch.tensor(x, requires_grad=True)


def zeros_like(x):
    return torch.zeros_like(x)


def abs(x):
    return torch.abs(x)
