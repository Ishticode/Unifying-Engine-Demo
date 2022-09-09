import Engine
import abc
import torch
from itertools import chain
Engine.choose_framework("torch")



class Module(abc.ABC):
    def __init__(self):
        pass
    # def get_params(self):
    #     self.params = [p.w for p in self.__dict__.values()] + [p.b for p in self.__dict__.values()]
    #     return self.params
    #
    # def set_params(self, params):
    #     self.params = params
    #     return self.params
    # def parameters(self):
    #     p = {"w": [d.w for d in self.__dict__.values()]}
    #     p["b"] = [d.b for d in self.__dict__.values()]
    #     return p

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):

        self.in_features = in_features
        self.out_features = out_features
        self.w = Engine.randn((out_features, in_features))
        self.w = Engine.variable(self.w)
        self.b = Engine.randn((out_features,)) if bias else None
        self.b = Engine.variable(self.b) if bias else None
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
        self._w_shape = filter_shape + [input_channels, output_channels] if data_format == 'NHWC' \
            else [input_channels, output_channels] + filter_shape
        self._b_shape = (1, 1, 1, output_channels)
        self.w = Engine.randn(self._w_shape)
        self.b = Engine.randn(self._b_shape)

    def parameters(self):
        return {"w":self.w, "b":self.b}

    def forward(self, inputs):
        return Engine.conv2d(inputs, self.w, self._strides, self._padding, self._data_format) + self.b


#
#
# class simpleNet(Module):
#     def __init__(self):
#         super(simpleNet, self).__init__()
#         self.conv1 = Conv2D(3, 32, [3, 3], 1, 'SAME')
#         self.conv2 = Conv2D(32, 64, [3, 3], 1, 'SAME')
#         self.conv3 = Conv2D(64, 128, [3, 3], 1, 'SAME')
#         self.fc1 = Linear(128, 64)
#         self.fc2 = Linear(64, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
#
#
# x = Engine.randn((2, 2, 4, 3))
# target = Engine.randn((2, 2, 4, 1))
# net = simpleNet()
# for param in net.__dict__.values():
#     (param.parameters())
# print(net.__dict__)
# grad  = Engine.grad_and_value(net, x, target)
# y = net(x)
# l = torch.nn.functional.cross_entropy(y, target)
# print(l)
#print(simpleNet()(x).shape)



















# lin = Linear(in_features=2, out_features=3)
# x = Engine.tensor([[1., 2], [3, 4]])
# tlin = torch.nn.functional.linear(x, lin.weight, lin.b)
# y = lin(x)
