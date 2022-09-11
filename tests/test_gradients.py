import Engine
import pytest
import numpy as np


class TestNet(Engine.Module):
    def __init__(self):
        self.linear1 = Engine.Linear(2, 2, bias=True, w=Engine.tensor([[1., 2], [3, 4]]), b=Engine.tensor([1., 2]))
        self.layers = [self.linear1]
        super(TestNet, self).__init__()

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        x = self.linear1(x)
        return x


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("x_input", [[[1., -2], [4, 5]]])
@pytest.mark.parametrize("true_grad_w", [[[20.5, 30.5],
       [57.5, 73.5]]])
@pytest.mark.parametrize("true_grad_b", [[ 4., 14.]])
def test_eval_and_grad(framework, x_input, true_grad_w, true_grad_b):
    Engine.choose_framework(framework)
    fn = TestNet()
    x_input, true_grad_w, true_grad_b = Engine.variable(x_input), Engine.tensor(true_grad_w), Engine.tensor(true_grad_b)
    target = x_input # target is irrelevant for this test but is a required arg by eval_and_grad
    loss, grad = Engine.eval_and_grad(fn, x_input, target, Engine.mse_loss)
    grad_w = grad[0]
    grad_b = grad[1]
    assert np.allclose(grad_w, true_grad_w)
    assert np.allclose(grad_b, true_grad_b)


# test gradient step
@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("x_input", [[[1., -2], [4, 5]]])
@pytest.mark.parametrize("target", [[[-1., -2], [4, 5]]])
def test_gradient_step(framework, x_input, target):
    Engine.choose_framework(framework)
    fn = TestNet()
    losses = []
    x_input = Engine.variable(x_input)
    target = Engine.tensor(target)
    for _ in range(5):
        loss, grad = Engine.eval_and_grad(fn, x_input, target, Engine.mse_loss)
        losses.append(loss)
        Engine.gradient_step(fn, grad)
    assert losses[0] > losses[2] > losses[4]
