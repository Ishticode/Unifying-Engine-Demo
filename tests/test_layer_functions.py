import pytest
import Engine
import numpy as np

@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
@pytest.mark.parametrize(
    "x_params_res", [
        ([[1., 2., 3.]],
         [[1., 1., 1.], [1., 1., 1.]],
         [2., 2.],
         [[8., 8.]]),
        ([[[1., 2., 3.]]],
         [[1., 1., 1.], [1., 1., 1.]],
         [2., 2.],
         [[[8., 8.]]])
    ])
def test_linear(framework, x_params_res):
    x, weight, bias, true_res = x_params_res
    Engine.choose_framework(framework)
    x, w, b, res = Engine.tensor(x), Engine.tensor(weight), Engine.tensor(bias), Engine.tensor(true_res)
    y = Engine.linear(x, w, b)
    assert np.allclose(y, res)

