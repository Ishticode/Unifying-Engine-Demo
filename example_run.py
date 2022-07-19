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
