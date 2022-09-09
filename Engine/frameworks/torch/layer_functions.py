import torch
import math

def linear(x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)


# noinspection PyUnresolvedReferences
def conv2d(x, filters, strides: int, padding: str, data_format: str = 'NHWC'):
    filter_shape = list(filters.shape[0:2])
    filters = filters.permute(3, 2, 0, 1)
    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)
    if padding == 'VALID':
        padding_list: List[int] = [0, 0]
    elif padding == 'SAME':
        padding_list: List[int] = [int(math.floor(item / 2)) for item in filter_shape]
    else:
        raise Exception('Invalid padding arg {}\n'
                        'Must be one of: "VALID" or "SAME"'.format(padding))
    res = torch.nn.functional.conv2d(x, filters, None, strides, padding_list)
    if data_format == 'NHWC':
        return res.permute(0, 2, 3, 1)
    return res
