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



# conv2d
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_res", [([[[[1.], [2.], [3.], [4.], [5.]],
                                   [[6.], [7.], [8.], [9.], [10.]],
                                   [[11.], [12.], [13.], [14.], [15.]],
                                   [[16.], [17.], [18.], [19.], [20.]],
                                   [[21.], [22.], [23.], [24.], [25.]]]],
                                 [[[[0.]], [[1.]], [[0.]]],
                                  [[[1.]], [[1.]], [[1.]]],
                                  [[[0.]], [[1.]], [[0.]]]],
                                 "SAME",
                                 [[[[9.], [13.], [17.], [21.], [19.]],
                                   [[25.], [35.], [40.], [45.], [39.]],
                                   [[45.], [60.], [65.], [70.], [59.]],
                                   [[65.], [85.], [90.], [95.], [79.]],
                                   [[59.], [83.], [87.], [91.], [69.]]]]),

                                ([[[[1.], [2.], [3.], [4.], [5.]],
                                   [[6.], [7.], [8.], [9.], [10.]],
                                   [[11.], [12.], [13.], [14.], [15.]],
                                   [[16.], [17.], [18.], [19.], [20.]],
                                   [[21.], [22.], [23.], [24.], [25.]]] for _ in range(5)],
                                 [[[[0.]], [[1.]], [[0.]]],
                                  [[[1.]], [[1.]], [[1.]]],
                                  [[[0.]], [[1.]], [[0.]]]],
                                 "SAME",
                                 [[[[9.], [13.], [17.], [21.], [19.]],
                                   [[25.], [35.], [40.], [45.], [39.]],
                                   [[45.], [60.], [65.], [70.], [59.]],
                                   [[65.], [85.], [90.], [95.], [79.]],
                                   [[59.], [83.], [87.], [91.], [69.]]] for _ in range(5)]),

                                ([[[[1.], [2.], [3.], [4.], [5.]],
                                   [[6.], [7.], [8.], [9.], [10.]],
                                   [[11.], [12.], [13.], [14.], [15.]],
                                   [[16.], [17.], [18.], [19.], [20.]],
                                   [[21.], [22.], [23.], [24.], [25.]]]],
                                 [[[[0.]], [[1.]], [[0.]]],
                                  [[[1.]], [[1.]], [[1.]]],
                                  [[[0.]], [[1.]], [[0.]]]],
                                 "VALID",
                                 [[[[35.], [40.], [45.]],
                                   [[60.], [65.], [70.]],
                                   [[85.], [90.], [95.]]]])])
@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
def test_conv2d(x_n_filters_n_pad_n_res, framework):
    Engine.choose_framework(framework)
    x, filters, padding, true_res = x_n_filters_n_pad_n_res
    x, filters, true_res = Engine.tensor(x), Engine.tensor(filters), Engine.tensor(true_res)
    ret = Engine.conv2d(x, filters, 1, padding)
    # type test
    assert str(type(ret)) == Engine.array_framework_classes[framework]
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(ret, true_res)


# max_pool
@pytest.mark.parametrize(
    "x", [[[[[7., 4],
          [9, 1]],

         [[3, 6],
          [9, 7]],

         [[0, 3],
          [5, 5]]],


        [[[9, 5],
          [8, 1]],

         [[8, 4],
          [4, 5]],

         [[6, 8],
          [1, 3]]]]])
@pytest.mark.parametrize("true_res", [[[[[9., 7.]]], [[[9., 5.]]]]])
@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
def test_max_pool(x,true_res, framework):
    Engine.choose_framework(framework)
    x, true_res = Engine.tensor(x), Engine.tensor(true_res)
    ret = Engine.max_pool(x, 2, 2)
    # type test
    assert str(type(ret)) == Engine.array_framework_classes[framework]
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(ret, true_res)




# relu
@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
@pytest.mark.parametrize("x", [[0., 2., -1., 3., -2.], [[0., 2., -1., 3., -2.]]])
@pytest.mark.parametrize("true_res", [[0., 2., 0., 3., 0.], [[0., 2., 0., 3., 0.]]])
def test_relu(framework, x, true_res):
    Engine.choose_framework(framework)
    x, true_res = Engine.tensor(x), Engine.tensor(true_res)
    ret = Engine.relu(x)
    # type test
    assert str(type(ret)) == Engine.array_framework_classes[framework]
    # value test
    assert np.allclose(ret, true_res)



