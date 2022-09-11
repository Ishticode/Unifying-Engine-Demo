import Engine

Engine.choose_framework("tensorflow")


class Network(Engine.Module):
    def __init__(self):
        self.linear1 = Engine.Linear(2, 3)
        self.linear2 = Engine.Linear(3, 2)
        self.layers = [self.linear1, self.linear2]
        super(Network, self).__init__()

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def loss_fn(y, target):
    return Engine.sum(Engine.abs(y - target))


x_in = Engine.variable([[1., 2], [3., 4]])
target = Engine.tensor([[4., 2], [2., 4]])
net = Network()

losses = []
epochs = []

for epoch in range(50):
    loss, grad = Engine.eval_and_grad(net, x_in, target, loss_fn)
    Engine.gradient_step(net, grad)
    losses.append(loss)
    epochs.append(epoch)


print(losses)
