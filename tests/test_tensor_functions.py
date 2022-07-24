import Engine
import pytest
import numpy as np

array_framework_classes = {
    "tensorflow": "<class 'tensorflow.python.framework.ops.EagerTensor'>",
    "numpy": "<class 'numpy.ndarray'>",
    "torch": "<class 'torch.Tensor'>",
    "chainer": "<class 'numpy.ndarray'>",
}


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



