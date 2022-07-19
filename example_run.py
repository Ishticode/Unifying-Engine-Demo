import Engine

frameworks_available = ["tensorflow", "numpy", "torch"]

for fw in frameworks_available:
    Engine.choose_framework(fw)
    x = Engine.tensor([[1, 2, 3], [4, 5, 6]])
    y = Engine.sum(x, 1)
    print(f"{fw}")
    print(x, type(x))
    print(y, type(y))
    print()

'''
tensorflow
tf.Tensor(
[[1 2 3]
 [4 5 6]], shape=(2, 3), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>
tf.Tensor([ 6 15], shape=(2,), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>

numpy
[[1 2 3]
 [4 5 6]] <class 'numpy.ndarray'>
[ 6 15] <class 'numpy.ndarray'>

torch
tensor([[1, 2, 3],
        [4, 5, 6]]) <class 'torch.Tensor'>
tensor([ 6, 15]) <class 'torch.Tensor'>'''