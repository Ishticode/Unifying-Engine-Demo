import torch
import math


def linear(x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)


def conv2d(x, filters, strides: int, padding: str, data_format: str = 'NHWC', dilations: int = 1):
    filter_shape = list(filters.shape[0:2])
    filters = filters.permute(3, 2, 0, 1)
    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)
    if padding == 'VALID':
        padding_list = [0, 0]
    elif padding == 'SAME':
        padding_list = [int(math.floor(item / 2)) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = torch.nn.functional.conv2d(x, filters, None, strides, padding_list, dilations)
    if data_format == 'NHWC':
        return res.permute(0, 2, 3, 1)
    return res


def relu(x):
    return torch.nn.functional.relu(x)


def max_pool(x, ksize, strides):
    x = x.permute(0, 3, 1, 2)
    res = torch.nn.functional.max_pool2d(x, ksize, strides, [0, 0])
    return res.permute(0, 2, 3, 1)


def mse_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)
