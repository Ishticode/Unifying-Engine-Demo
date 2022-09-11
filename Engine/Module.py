import Engine
import abc
import torch
from itertools import chain



class Module(abc.ABC):

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, w=None, b=None):

        self.in_features = in_features
        self.out_features = out_features

        if w is None:
            self.w = Engine.randn((out_features, in_features))
        else:
            self.w = w

        if b is None:
            if bias:
                self.b = Engine.randn((out_features,))
            else:
                self.b = None
        else:
            self.b = b

        self.w = Engine.variable(self.w)
        self.b = Engine.variable(self.b)
        super(Linear, self).__init__()

    def forward(self, x):
        return Engine.linear(x, self.w, self.b)

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x):
        return self.forward(x)


class Conv2D(Module):

    def __init__(self, input_channels, output_channels, filter_shape, strides, padding,
                 data_format='NHWC'):

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filter_shape = filter_shape
        self._strides = strides
        self._padding = padding
        self._data_format = data_format
        self._w_shape = list(filter_shape) + [input_channels, output_channels] if data_format == 'NHWC' \
            else [input_channels, output_channels] + list(filter_shape)
        self._b_shape = (1, 1, 1, output_channels)
        self.w = Engine.randn(self._w_shape)
        self.w = Engine.variable(self.w)
        self.b = Engine.randn(self._b_shape)
        self.b = Engine.variable(self.b)

    def parameters(self):
        return [self.w, self.b]

    def forward(self, inputs):
        return Engine.conv2d(inputs, self.w, self._strides, self._padding) + self.b


class MaxPool2D(Module):

    def __init__(self, ksize, strides):
        self._ksize = ksize
        self._strides = strides

    def forward(self, inputs):
        return Engine.max_pool(inputs, self._ksize, self._strides)

#
#



















# lin = Linear(in_features=2, out_features=3)
# x = Engine.tensor([[1., 2], [3, 4]])
# tlin = torch.nn.functional.linear(x, lin.weight, lin.b)
# y = lin(x)
