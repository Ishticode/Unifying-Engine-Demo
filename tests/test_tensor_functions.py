import Engine
import pytest
import numpy as np
from Engine import array_framework_classes



@pytest.mark.parametrize("framework", ["tensorflow", "numpy", "torch", "chainer"])
@pytest.mark.parametrize("x", [[[1, 2, 3], [4, 5, 6]], [[2., 3.]]])
def test_tensor(framework, x):
    Engine.choose_framework(framework)
    y = Engine.tensor(x)
    assert str(type(y)) == array_framework_classes[framework]


@pytest.mark.parametrize("framework", ["tensorflow", "numpy", "torch", "chainer"])
@pytest.mark.parametrize("x_input", [[[1, 2, 3], [4, 5, 6]], [[2., 3.],[4., 5.]]])
@pytest.mark.parametrize("axis", [0, 1])
def test_sum(framework, x_input, axis):
    Engine.choose_framework(framework)
    x = Engine.tensor(x_input)
    y = Engine.sum(x, axis)

    # test array type is correct
    assert str(type(y)) == array_framework_classes[framework]

    # test results against numpy
    np_result = np.sum(x_input, axis)
    assert np.allclose(y, np_result)


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("shape", [(2, 3), (2,), (1, 1)])
def test_randn(framework, shape):
    Engine.choose_framework(framework)
    x = Engine.randn(shape)
    assert str(type(x)) == array_framework_classes[framework]
    assert x.shape == shape


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("x_input", [[[1, 2, 3], [4, 5, 6]], [[2., 3.],[4., 5.]]])
def test_zeros_like(framework, x_input):
    Engine.choose_framework(framework)
    x_input = Engine.tensor(x_input)
    y = Engine.zeros_like(x_input)
    assert str(type(y)) == array_framework_classes[framework]
    assert y.shape == x_input.shape


def test_variable_torch():
    Engine.choose_framework("torch")
    x = Engine.tensor([1., 2., 3.])
    y = Engine.variable(x)
    assert str(type(y)) == array_framework_classes["torch"]
    assert y.requires_grad
    assert str(y.dtype) == "torch.float32"


def test_variable_tensorflow():
    Engine.choose_framework("tensorflow")
    x = Engine.tensor([1., 2., 3.])
    y = Engine.variable(x)
    assert str(type(y)) == "<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>"
    assert y.numpy().dtype == "float32"


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("x_input", [[[1, -2, 3], [4, 5, 6]], [[2., -3.],[4., 5.]]])
def test_abs(framework, x_input):
    Engine.choose_framework(framework)
    x_input = Engine.tensor(x_input)
    y = Engine.abs(x_input)
    assert str(type(y)) == array_framework_classes[framework]
    assert y.shape == x_input.shape
    x_np = np.array(x_input)
    assert np.allclose(y, np.abs(x_np))


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("x_input", [[[1, -2, 3], [4, 5, 6]]])
@pytest.mark.parametrize("shape", [(2, 3), (6,), (6, 1)])
def test_reshape(framework, x_input, shape):
    Engine.choose_framework(framework)
    x_input = Engine.tensor(x_input)
    y = Engine.reshape(x_input, shape)
    assert str(type(y)) == array_framework_classes[framework]
    assert y.shape == shape
    x_np = np.array(x_input)
    assert np.allclose(y, np.reshape(x_np, shape))

