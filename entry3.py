import Engine
from itertools import chain
Engine.choose_framework("tensorflow")
import torch
import tensorflow as tf

def loss_fn(y, target):
    return Engine.sum(Engine.abs(y - target))


class network(Engine.Module):
    def __init__(self):
        self.linear1 = Engine.Linear(2, 3)
        self.linear2 = Engine.Linear(3, 2)
        self.layers = [self.linear1, self.linear2]
        super(network, self).__init__()

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


x_in = Engine.variable([[1., 2], [3., 4]])
target = Engine.tensor([[4., 2], [2., 4]])
net = network()

# new_params = []
#
# for p, g in zip(net.parameters(), grads):
#     p = Engine.stop_gradient(p)
#     print(p)
#     p.assign(tf.subtract(p, 0.001*g))
#     print(p)
#     new_params.append(p)
# [p.assign(n) for p, n in zip(net.parameters(), new_params)]

# print("parameters after update")
# print(net.parameters())

# print(net.parameters())
# p = [[d.w, d.b] for d in net.__dict__.values()]
# p = list(chain.from_iterable(p))
# # p = p[0] + p[1]
# print(p)
# print(len(p))

#print(list(net.parameters().values()))


# losses = []
# epochs = []
# # new_params = []
# for epoch in range(10):
#     loss, grad = Engine.eval_and_grad(net, x_in, target, loss_fn)
#
#     for p, g in zip(net.parameters(), grad):
#         p = Engine.stop_gradient(p)
#         p -= 0.001 * g
#         g = None
#     losses.append(loss)
#     epochs.append(epoch)
#
# print(losses)

losses = []
epochs = []
# new_params = []
for epoch in range(50):
    loss, grad = Engine.eval_and_grad(net, x_in, target, loss_fn)
    Engine.gradient_step(net, grad)
    losses.append(loss)
    epochs.append(epoch)
    #grad = None

print(losses)