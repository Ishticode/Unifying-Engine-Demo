import Engine
from itertools import chain
Engine.choose_framework("torch")
import torch
import tensorflow as tf

loss_fn = Engine.mse_loss

class network(Engine.Module):
    def __init__(self):
        self.conv1 = Engine.Conv2D(1, 4, (3, 3), (1, 1), 'SAME')
        self.conv2 = Engine.Conv2D(4, 8, (3, 3), (1, 1), 'SAME')
        self.pool = Engine.MaxPool2D((2, 2), (2, 2))
        self.layers = [self.conv1, self.conv2]
        super(network, self).__init__()

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = Engine.relu(x)
        return x


x_in = Engine.variable(Engine.randn((2,2,2,1)))
target = Engine.randn((2,2,2,1))
net = network()

losses = []
epochs = []

for epoch in range(30):
    loss, grad = Engine.eval_and_grad(net, x_in, target, loss_fn)
    Engine.gradient_step(net, grad)
    losses.append(loss)
    epochs.append(epoch)

print(losses)