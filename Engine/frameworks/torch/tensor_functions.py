import torch


def tensor(x):
    return torch.tensor(x)


def sum(x, axis=None):
    return torch.sum(x, dim=axis)
